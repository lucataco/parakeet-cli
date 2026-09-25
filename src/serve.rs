mod collection;
pub(crate) mod protocol;
mod runtime;
mod worker;

use crate::{
    DAEMON_PROTOCOL_VERSION,
    segments::{PREVIEW_MAX_OWNED_SAMPLES, Preview, PreviewCadence, Segmenter},
    session_recognition::{Recognizer, SessionResult},
    vad,
};
use anyhow::{Context, Result};
use collection::{CaptureEnd, Collected};
use crossbeam_channel::{Receiver, Sender};
use protocol::Command;
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, Ordering},
    },
    time::Duration,
};
use tokio::{io::AsyncWriteExt, net::UnixStream, sync::Notify};

const RUN: u8 = 0;
const STOP: u8 = 1;
const CANCEL: u8 = 2;

struct ActiveSession {
    id: String,
    phase: &'static str,
    control: Arc<AtomicU8>,
}
type State = Arc<Mutex<Option<ActiveSession>>>;

struct Job {
    id: String,
    control: Arc<AtomicU8>,
    receiver: Receiver<Collected>,
    /// Interim previews; its sender is dropped at once when a session did not
    /// ask for partials, so the worker only ever waits on segments.
    previews: Receiver<Preview>,
    response: Option<tokio::sync::oneshot::Sender<Value>>,
}

#[derive(Clone)]
struct DaemonContext {
    state: State,
    jobs: Sender<Job>,
    device: Option<String>,
    shutdown: Arc<Notify>,
    vad_path: PathBuf,
}

impl DaemonContext {
    async fn handle(&self, mut request: Command) -> Result<Value> {
        if request.command == "toggle" {
            let active = self
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("Session state failed"))?;
            request.command = if active.is_none() { "start" } else { "stop" }.into();
        }
        if request.command == "start" || request.command == "capture" {
            self.start_session(request).await
        } else {
            self.control_session(request)
        }
    }

    async fn start_session(&self, request: Command) -> Result<Value> {
        let capture_mode = request.command == "capture";
        let capture_end = if capture_mode {
            Some(CaptureEnd::load(&self.vad_path)?)
        } else {
            None
        };
        let id = request.session_id.unwrap_or_else(new_session_id);
        anyhow::ensure!(!id.is_empty() && id.len() <= 128, "Invalid session_id");
        let control = Arc::new(AtomicU8::new(RUN));
        {
            let mut active = self
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("Session state failed"))?;
            anyhow::ensure!(active.is_none(), "A recording is still pending completion");
            *active = Some(ActiveSession {
                id: id.clone(),
                phase: "recording",
                control: control.clone(),
            });
        }
        let (tx, receiver) = crossbeam_channel::bounded(32);
        // One slot is enough: the worker always takes the newest preview and a
        // missed tick is simply skipped.
        let (preview_tx, previews) = crossbeam_channel::bounded(1);
        let preview_tx = (request.partials && !capture_mode).then_some(preview_tx);
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        if self
            .jobs
            .send(Job {
                id: id.clone(),
                control: control.clone(),
                receiver,
                previews,
                response: capture_mode.then_some(response_tx),
            })
            .is_err()
        {
            *self
                .state
                .lock()
                .map_err(|_| anyhow::anyhow!("Session state failed"))? = None;
            anyhow::bail!("Recognition worker unavailable");
        }
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (device, job_id, capture_control) = (self.device.clone(), id.clone(), control.clone());
        std::thread::spawn(move || {
            collection::collect(
                device,
                job_id,
                capture_control,
                tx,
                preview_tx,
                started_tx,
                capture_end,
            )
        });
        started_rx
            .await
            .context("Capture thread exited")?
            .map_err(anyhow::Error::msg)?;
        if capture_mode {
            return match tokio::time::timeout(Duration::from_secs(60), response_rx).await {
                Ok(value) => Ok(value.context("Capture worker exited")?),
                Err(_) => {
                    control.store(CANCEL, Ordering::SeqCst);
                    anyhow::bail!("Capture timed out");
                }
            };
        }
        Ok(protocol::acknowledged("recording", Some(&id)))
    }

    fn control_session(&self, request: Command) -> Result<Value> {
        let mut active = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Session state failed"))?;
        match request.command.as_str() {
            "stop" | "cancel" => {
                let session = active.as_mut().context("No active recording")?;
                anyhow::ensure!(
                    request
                        .session_id
                        .as_deref()
                        .is_none_or(|id| id == session.id),
                    "Session identifier does not match"
                );
                session.phase = "transcribing";
                if request.command == "cancel" {
                    session.control.store(CANCEL, Ordering::SeqCst);
                } else {
                    let _ = session.control.compare_exchange(
                        RUN,
                        STOP,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    );
                }
            }
            "status" => {}
            "shutdown" => {
                if let Some(session) = active.as_ref() {
                    session.control.store(CANCEL, Ordering::SeqCst);
                }
                self.shutdown.notify_one();
            }
            _ => anyhow::bail!("Unknown command"),
        }
        Ok(protocol::acknowledged(
            active.as_ref().map_or("idle", |session| session.phase),
            active.as_ref().map(|session| session.id.as_str()),
        ))
    }
}

fn new_session_id() -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

async fn connection(mut stream: UnixStream, context: DaemonContext) -> Result<()> {
    let bytes = protocol::read_command(&mut stream).await?;
    let response = match protocol::parse_command(&bytes) {
        Ok(request) => context.handle(request).await,
        Err(error) => Err(error.into()),
    }
    .unwrap_or_else(protocol::error);
    stream.write_all(format!("{response}\n").as_bytes()).await?;
    Ok(())
}

pub async fn run_serve(
    socket: &Path,
    pid: &Path,
    device: &Option<String>,
    model_dir: &Path,
    clipboard: bool,
    _verbose: bool,
    coreml: bool,
) -> Result<()> {
    let vad_path = vad::ensure_vad_model(model_dir).await?;
    let recognizer = Recognizer::load(model_dir, &vad_path, coreml)?;
    let (_files, listener) = runtime::RuntimeFiles::bind(socket, pid).await?;
    let state = State::default();
    let (jobs, receiver) = crossbeam_channel::bounded(1);
    let worker_state = state.clone();
    std::thread::spawn(move || worker::run(recognizer, receiver, worker_state, clipboard));
    let context = DaemonContext {
        state,
        jobs,
        device: device.clone(),
        shutdown: Arc::new(Notify::new()),
        vad_path,
    };
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut toggle_signal =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined1())?;
    let mut stop_signal =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined2())?;
    eprintln!("[daemon] Ready. Waiting for commands (protocol {DAEMON_PROTOCOL_VERSION})...");
    loop {
        tokio::select! {
            _ = context.shutdown.notified() => break,
            _ = terminate.recv() => break,
            _ = tokio::signal::ctrl_c() => break,
            _ = toggle_signal.recv() => { let _ = context.handle(Command::bare("toggle")).await; }
            _ = stop_signal.recv() => { let _ = context.handle(Command::bare("stop")).await; }
            accepted = listener.accept() => {
                let (stream, _) = accepted?;
                let client_context = context.clone();
                tokio::spawn(async move {
                    if let Err(error) = connection(stream, client_context).await { eprintln!("Socket error: {error}"); }
                });
            }
        }
    }
    if let Some(session) = context
        .state
        .lock()
        .map_err(|_| anyhow::anyhow!("Session state failed"))?
        .as_ref()
    {
        session.control.store(CANCEL, Ordering::SeqCst);
    }
    Ok(())
}

/// Replays a file through the session pipeline. With `partials`, the same
/// preview cadence the daemon uses emits `partial` events to stdout as the
/// audio "arrives", so interim output can be checked deterministically
/// without a microphone. The final result is unaffected either way.
pub async fn replay(
    file: &Path,
    model_dir: &Path,
    coreml: bool,
    partials: bool,
) -> Result<SessionResult> {
    let vad_path = vad::ensure_vad_model(model_dir).await?;
    let mut recognizer = Recognizer::load(model_dir, &vad_path, coreml)?;
    let samples = crate::audio::load_wav_file(file, false)?;
    let mut segmenter = Segmenter::default();
    let mut cadence = PreviewCadence::default();
    let mut emitter = protocol::PartialEmitter::default();
    let mut result = SessionResult::default();
    for chunk in samples.chunks(480) {
        for segment in segmenter.push(chunk) {
            recognizer.append(segment, &mut result);
        }
        if partials && cadence.observe(chunk.len(), segmenter.owned_pending()) {
            if let Some(preview) = segmenter.preview(PREVIEW_MAX_OWNED_SAMPLES) {
                let truncated = preview.truncated;
                let audio_ms = protocol::samples_to_ms(preview.audio_samples);
                if let Some(tokens) = recognizer.preview(preview.segment) {
                    let mut running = result.tokens.clone();
                    running.extend(tokens);
                    let text = recognizer.decode(&running);
                    if let Some(event) = emitter.next("replay", text, truncated, audio_ms) {
                        protocol::emit(event);
                    }
                }
            }
        }
    }
    if let Some(segment) = segmenter.finish() {
        recognizer.append(segment, &mut result);
    }
    result.text = recognizer.decode(&result.tokens);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::StreamingResampler;
    use collection::{enqueue_segment, enqueue_segments, finish_audio};

    #[test]
    fn stop_drains_queued_audio_and_resampler_tail() {
        let (capture_tx, capture_rx) = crossbeam_channel::bounded(2);
        let (tx, rx) = crossbeam_channel::bounded(2);
        let mut resampler = StreamingResampler::new(48000, 16000);
        let mut segments = Segmenter::default();
        let mut dropped = 0;
        let input: Vec<f32> = (0..5000).map(|i| i as f32 / 5000.0).collect();
        enqueue_segments(
            &mut segments,
            &resampler.process(&input[..1111]),
            &tx,
            &mut dropped,
        );
        capture_tx.send(input[1111..4000].to_vec()).unwrap();
        capture_tx.send(input[4000..].to_vec()).unwrap();
        drop(capture_tx);
        finish_audio(
            &capture_rx,
            &mut resampler,
            &mut segments,
            &tx,
            &mut dropped,
        );
        let Collected::Segment(segment) = rx.recv().unwrap() else {
            panic!("missing tail")
        };
        let expected = crate::audio::resample::resample_linear(&input, 48000, 16000);
        assert_eq!(
            segment.samples.len(),
            expected.len() + crate::segments::PREVIEW_TAIL_SILENCE_SAMPLES
        );
        for (actual, expected) in segment.samples.iter().zip(&expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!(
            segment.samples[expected.len()..]
                .iter()
                .all(|sample| *sample == 0.0)
        );
        assert_eq!(dropped, 0);
    }

    #[tokio::test]
    async fn pending_session_rejects_start_and_mismatched_stop() {
        let control = Arc::new(AtomicU8::new(RUN));
        let state = Arc::new(Mutex::new(Some(ActiveSession {
            id: "original".into(),
            phase: "transcribing",
            control: control.clone(),
        })));
        let (jobs, _receiver) = crossbeam_channel::bounded(1);
        let context = DaemonContext {
            state,
            jobs,
            device: None,
            shutdown: Arc::new(Notify::new()),
            vad_path: PathBuf::new(),
        };
        for name in ["start", "stop", "cancel"] {
            assert!(
                context
                    .handle(Command {
                        command: name.into(),
                        session_id: Some("new".into()),
                        partials: false,
                    })
                    .await
                    .is_err()
            );
            assert_eq!(
                context.state.lock().unwrap().as_ref().unwrap().id,
                "original"
            );
        }
        for name in ["cancel", "stop"] {
            context
                .handle(Command {
                    command: name.into(),
                    session_id: Some("original".into()),
                    partials: false,
                })
                .await
                .unwrap();
        }
        assert_eq!(control.load(Ordering::SeqCst), CANCEL);
    }

    #[test]
    fn previews_are_offered_on_cadence_only_when_requested_and_never_count_as_drops() {
        use crate::segments::{PREVIEW_INTERVAL_SAMPLES, PreviewCadence};
        use collection::offer_preview;
        let (segment_tx, segment_rx) = crossbeam_channel::bounded(32);
        let (preview_tx, preview_rx) = crossbeam_channel::bounded(1);
        let mut segmenter = Segmenter::default();
        let mut cadence = PreviewCadence::default();
        let mut dropped = 0;
        let mut offered = 0;
        let chunk = vec![0.3; 1_600];
        for _ in 0..20 {
            enqueue_segments(&mut segmenter, &chunk, &segment_tx, &mut dropped);
            if offer_preview(&mut cadence, &segmenter, chunk.len(), Some(&preview_tx)) {
                offered += 1;
            }
        }
        // Two seconds of audio: previews at 0.5, 1.0, 1.5 and 2.0 s reach the
        // slot, but each later one waits behind the first until the worker
        // drains it.
        assert_eq!(offered, 1);
        assert_eq!(
            20 * 1_600 / PREVIEW_INTERVAL_SAMPLES,
            4,
            "cadence would have fired four times"
        );
        let preview = preview_rx.recv().unwrap();
        assert!(!preview.truncated);
        // The first tick fires on the chunk that crosses the interval: 5 × 1600
        // samples, plus the silent run-out every preview carries.
        assert_eq!(
            preview.segment.owned_end - preview.segment.owned_start,
            5 * 1_600 + crate::segments::PREVIEW_TAIL_SILENCE_SAMPLES
        );
        assert_eq!(preview.audio_samples, 5 * 1_600);
        assert!(
            segment_rx.try_recv().is_err(),
            "no committed segment below one core span"
        );
        assert_eq!(dropped, 0, "a full preview slot is never loss");

        let mut silent = PreviewCadence::default();
        for _ in 0..20 {
            assert!(!offer_preview(&mut silent, &segmenter, chunk.len(), None));
        }
        assert!(preview_rx.try_recv().is_err());
    }

    #[test]
    fn collection_keeps_running_during_slow_inference_and_reports_overrun() {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let mut segmenter = Segmenter::default();
        let mut dropped = 0;
        for _ in 0..90 {
            enqueue_segments(&mut segmenter, &[0.1; 16000], &tx, &mut dropped);
        }
        let final_segment = segmenter.finish().unwrap();
        let tail_count = final_segment.owned_end
            - final_segment.owned_start
            - crate::segments::PREVIEW_TAIL_SILENCE_SAMPLES;
        let Collected::Segment(first) = rx.recv().unwrap() else {
            panic!("missing first segment")
        };
        assert_eq!(
            first.owned_end - first.owned_start + dropped as usize + tail_count,
            90 * 16000
        );
        assert!(dropped > 0);
        enqueue_segment(final_segment, &tx, &mut dropped);
        assert!(matches!(rx.recv().unwrap(), Collected::Segment(_)));
    }
}
