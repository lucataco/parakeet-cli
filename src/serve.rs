mod collection;
pub(crate) mod protocol;
mod runtime;
mod worker;

use crate::{
    DAEMON_PROTOCOL_VERSION,
    segments::{PREVIEW_MAX_OWNED_SAMPLES, Preview, Segmenter, Ticker},
    session_recognition::{Recognizer, SessionResult},
    vad::{self, SilenceMeter},
};
use anyhow::{Context, Result};
use collection::{Attach, CaptureEnd, Collected, Interim, WarmSlot};
use crossbeam_channel::{Receiver, Sender};
use protocol::Command;
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU8, Ordering},
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
    keep_warm: Arc<AtomicBool>,
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
    warm: WarmSlot,
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
        let (preview_tx, previews) = crossbeam_channel::bounded(1);
        let interim = if request.partials && !capture_mode {
            Some(Interim {
                ticker: Ticker::new(load_silence(&self.vad_path)),
                previews: preview_tx,
            })
        } else {
            None
        };
        let id = request.session_id.unwrap_or_else(new_session_id);
        anyhow::ensure!(!id.is_empty() && id.len() <= 128, "Invalid session_id");
        let control = Arc::new(AtomicU8::new(RUN));
        let keep_warm = Arc::new(AtomicBool::new(false));
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
                keep_warm: keep_warm.clone(),
            });
        }
        let (tx, receiver) = crossbeam_channel::bounded(32);
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
        let attach = Attach {
            job_id: id.clone(),
            control: control.clone(),
            keep_warm,
            tx,
            interim,
            started: started_tx,
            capture_end,
        };
        // A microphone kept warm by the previous session already holds the
        // audio spoken since it stopped; hand the session to it.
        let warm_sender = self.warm.lock().ok().and_then(|mut slot| slot.take());
        let attach = match warm_sender {
            Some(sender) => match sender.send(attach) {
                Ok(()) => None,
                Err(unsent) => Some(unsent.into_inner()),
            },
            None => Some(attach),
        };
        if let Some(attach) = attach {
            let (device, warm) = (self.device.clone(), self.warm.clone());
            std::thread::spawn(move || collection::collect(device, attach, warm));
        }
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
                if request.command == "stop" && request.keep_warm {
                    session.keep_warm.store(true, Ordering::SeqCst);
                }
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

/// Environment variable naming the process that launched the daemon. When set,
/// the daemon shuts down (removing its socket and pid file) once that process
/// is gone, so a crashed or force-quit client can't leave the model and the
/// microphone held. Ignored by older daemons, so clients can always set it.
pub const PARENT_PID_ENV: &str = "PARAKEET_PARENT_PID";

fn watch_parent(shutdown: &Arc<Notify>) {
    let Some(parent) = std::env::var(PARENT_PID_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
    else {
        return;
    };
    let shutdown = shutdown.clone();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(1));
            if !parent_is_alive(parent, std::os::unix::process::parent_id()) {
                eprintln!("[daemon] Parent process {parent} exited; shutting down");
                shutdown.notify_one();
                return;
            }
        }
    });
}

/// A process whose parent exits is re-parented (to launchd), so its parent id
/// no longer matches the one it was started with.
fn parent_is_alive(expected: u32, current_parent: u32) -> bool {
    expected == current_parent
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
    let (vad_path, recognizer) = load_recognizer_timed(model_dir, coreml).await?;
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
        warm: WarmSlot::default(),
    };
    watch_parent(&context.shutdown);
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

async fn load_recognizer_timed(model_dir: &Path, coreml: bool) -> Result<(PathBuf, Recognizer)> {
    let started = std::time::Instant::now();
    let vad_path = vad::ensure_vad_model(model_dir).await?;
    let checked = started.elapsed();
    let recognizer = Recognizer::load(model_dir, &vad_path, coreml)?;
    eprintln!(
        "Model loaded in {:.3}s (VAD check {:.3}s, recognizer {:.3}s)",
        started.elapsed().as_secs_f64(),
        checked.as_secs_f64(),
        (started.elapsed() - checked).as_secs_f64()
    );
    Ok((vad_path, recognizer))
}

fn load_silence(vad_path: &Path) -> Option<SilenceMeter> {
    SilenceMeter::silero(vad_path)
        .inspect_err(|error| eprintln!("Silence VAD unavailable; no ticks this session: {error:#}"))
        .ok()
}

/// Replays a file through the session pipeline. With `partials`, the same
/// preview cadence the daemon uses emits `tick` and `partial` events to stdout
/// as the audio "arrives", so interim output can be checked deterministically
/// without a microphone. The final result is unaffected either way.
pub async fn replay(
    file: &Path,
    model_dir: &Path,
    coreml: bool,
    partials: bool,
) -> Result<SessionResult> {
    let (vad_path, mut recognizer) = load_recognizer_timed(model_dir, coreml).await?;
    let samples = crate::audio::load_wav_file(file, false)?;
    let mut segmenter = Segmenter::default();
    let mut ticker = partials.then(|| Ticker::new(load_silence(&vad_path)));
    let mut emitter = protocol::PartialEmitter::default();
    let mut result = SessionResult::default();
    for chunk in samples.chunks(480) {
        for segment in segmenter.push(chunk) {
            recognizer.append(segment, &mut result);
        }
        let Some(tick) = ticker.as_mut().and_then(|t| t.observe(&segmenter, chunk)) else {
            continue;
        };
        if let Some(event) = protocol::tick("replay", tick) {
            protocol::emit(event);
        }
        if let Some(preview) = segmenter.preview(PREVIEW_MAX_OWNED_SAMPLES) {
            worker::emit_partial(&mut recognizer, &result, "replay", preview, &mut emitter);
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
    use crate::vad::VAD_CHUNK_SAMPLES;
    use serde_json::json;

    fn loud_is_speech() -> SilenceMeter {
        SilenceMeter::new(|chunk| {
            Ok(if chunk.iter().any(|sample| sample.abs() >= 0.1) {
                0.9
            } else {
                0.1
            })
        })
    }

    fn hear_speech(
        interim: &mut Interim,
        previews: &Receiver<Preview>,
        chunks: usize,
    ) -> Vec<(Option<Value>, bool)> {
        let mut segmenter = Segmenter::default();
        let mut steps = Vec::new();
        for _ in 0..chunks {
            let chunk = vec![0.3; 1_600];
            segmenter.push(&chunk);
            interim.observe(&segmenter, &chunk, |tick| {
                steps.push((protocol::tick("s", tick), false));
            });
            if previews.try_recv().is_ok() {
                if let Some(step) = steps.last_mut() {
                    step.1 = true;
                }
            }
        }
        steps
    }

    #[test]
    fn parent_watch_detects_re_parenting() {
        assert!(parent_is_alive(4242, 4242));
        assert!(!parent_is_alive(4242, 1));
    }
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
            keep_warm: Arc::new(AtomicBool::new(false)),
        })));
        let (jobs, _receiver) = crossbeam_channel::bounded(1);
        let context = DaemonContext {
            state,
            jobs,
            device: None,
            shutdown: Arc::new(Notify::new()),
            vad_path: PathBuf::new(),
            warm: WarmSlot::default(),
        };
        for name in ["start", "stop", "cancel"] {
            assert!(
                context
                    .handle(Command {
                        command: name.into(),
                        session_id: Some("new".into()),
                        partials: false,
                        keep_warm: false,
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
                    keep_warm: true,
                })
                .await
                .unwrap();
        }
        assert_eq!(control.load(Ordering::SeqCst), CANCEL);
        assert!(
            context
                .state
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .keep_warm
                .load(Ordering::SeqCst),
            "stop with keep_warm marks the session"
        );
    }

    #[test]
    fn a_session_without_a_silence_meter_still_offers_previews_and_sends_no_ticks() {
        let (preview_tx, previews) = crossbeam_channel::bounded(1);
        let mut interim = Interim {
            ticker: Ticker::new(None),
            previews: preview_tx,
        };
        let steps = hear_speech(&mut interim, &previews, 20);
        assert_eq!(steps.len(), 4, "the cadence still fires every 0.5 s");
        assert!(
            steps
                .iter()
                .all(|(tick, previewed)| tick.is_none() && *previewed)
        );
    }

    #[test]
    fn a_missing_vad_model_leaves_the_session_without_a_meter() {
        assert!(load_silence(Path::new("/nonexistent/silero_vad.onnx")).is_none());
    }

    #[test]
    fn a_vad_failure_mid_session_stops_ticks_but_not_previews() {
        let (preview_tx, previews) = crossbeam_channel::bounded(1);
        let mut chunks_seen = 0;
        let failing_after_one_second = SilenceMeter::new(move |_| {
            chunks_seen += 1;
            anyhow::ensure!(chunks_seen <= 31, "model failed");
            Ok(0.9)
        });
        let mut interim = Interim {
            ticker: Ticker::new(Some(failing_after_one_second)),
            previews: preview_tx,
        };
        let steps = hear_speech(&mut interim, &previews, 30);
        let ticked: Vec<_> = steps.iter().map(|(tick, _)| tick.is_some()).collect();
        assert_eq!(
            ticked,
            [true, true, false, false, false, false],
            "ticks stop after the chunk the model failed on, never counting it as silence"
        );
        assert!(steps.iter().all(|(_, previewed)| *previewed));
    }

    #[test]
    fn every_cadence_tick_reports_silence_while_a_busy_worker_misses_previews() {
        let (segment_tx, segment_rx) = crossbeam_channel::bounded(32);
        let (preview_tx, preview_rx) = crossbeam_channel::bounded(1);
        let mut interim = Interim {
            ticker: Ticker::new(Some(loud_is_speech())),
            previews: preview_tx,
        };
        let mut segmenter = Segmenter::default();
        let mut dropped = 0;
        let mut events = Vec::new();
        for level in [0.3; 10].into_iter().chain([0.0; 10]) {
            let chunk = vec![level; 1_600];
            enqueue_segments(&mut segmenter, &chunk, &segment_tx, &mut dropped);
            interim.observe(&segmenter, &chunk, |tick| {
                assert!(
                    !events.is_empty() || preview_rx.is_empty(),
                    "the first tick is reported before its preview is offered"
                );
                events.extend(protocol::tick("s", tick));
            });
        }
        let speech_chunk_end = 16_000_usize.next_multiple_of(VAD_CHUNK_SAMPLES);
        let whole_chunks = |received: usize| received - received % VAD_CHUNK_SAMPLES;
        let silence_ms =
            |received: usize| protocol::samples_to_ms(whole_chunks(received) - speech_chunk_end);
        let reported: Vec<_> = events
            .iter()
            .map(|event| {
                assert_eq!(event["type"], "tick");
                assert_eq!(event["session_id"], "s");
                (
                    event["audio_ms"].clone(),
                    event["trailing_silence_ms"].clone(),
                )
            })
            .collect();
        assert_eq!(
            reported,
            [
                (500, 0),
                (1_000, 0),
                (1_500, silence_ms(24_000)),
                (2_000, silence_ms(32_000))
            ]
            .map(|(audio, silence)| (json!(audio), json!(silence)))
        );
        let preview = preview_rx.recv().unwrap();
        assert!(
            preview_rx.try_recv().is_err(),
            "the worker never drained the slot, so later previews were skipped"
        );
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
