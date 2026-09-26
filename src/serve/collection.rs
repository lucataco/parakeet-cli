use super::{RUN, protocol};
use crate::{
    audio::StreamingResampler,
    segments::{PREVIEW_MAX_OWNED_SAMPLES, Preview, PreviewCadence, Segment, Segmenter},
    session_capture::Capture,
    vad::{SileroVad, VAD_CHUNK_SAMPLES, VadEvent, VadSegmenter},
};
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use std::{
    collections::VecDeque,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
    time::{Duration, Instant},
};

/// Longest stretch of audio a warm microphone keeps for the next session (3 s).
pub(super) const WARM_PREROLL_SECONDS: usize = 3;
/// How long a warm microphone waits for the next `start` before closing (5 s).
pub(super) const WARM_TIMEOUT: Duration = Duration::from_secs(5);

/// Everything a capture thread needs to record one session.
pub(super) struct Attach {
    pub job_id: String,
    pub control: Arc<AtomicU8>,
    /// Set by `stop` with `"keep_warm": true`: after this session the capture
    /// thread keeps the microphone open and hands its audio to the next one.
    pub keep_warm: Arc<AtomicBool>,
    pub tx: Sender<Collected>,
    pub previews: Option<Sender<Preview>>,
    pub started: tokio::sync::oneshot::Sender<std::result::Result<(), String>>,
    pub capture_end: Option<CaptureEnd>,
}

/// Where a warm capture thread waits for its next session. `start` takes the
/// sender out; the thread only closes the microphone after clearing it, under
/// the same lock, so a taken sender always reaches a live thread.
pub(super) type WarmSlot = Arc<Mutex<Option<Sender<Attach>>>>;

/// The newest audio heard between sessions, bounded so a long gap keeps only
/// its last few seconds.
pub(super) struct PrerollRing {
    samples: VecDeque<f32>,
    capacity: usize,
}

impl PrerollRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        self.samples.extend(samples.iter().copied());
        let excess = self.samples.len().saturating_sub(self.capacity);
        self.samples.drain(..excess);
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.samples.into()
    }
}

pub(super) enum Collected {
    Segment(Segment),
    Done { dropped: u64, error: Option<String> },
}

pub(super) struct CaptureEnd {
    vad: SileroVad,
    segmenter: VadSegmenter,
    pending: Vec<f32>,
}

impl CaptureEnd {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self {
            vad: SileroVad::load(path, false)?,
            segmenter: VadSegmenter::new(0.5, 1500),
            pending: Vec::new(),
        })
    }

    fn process(&mut self, samples: &[f32]) -> Result<bool> {
        self.pending.extend_from_slice(samples);
        while self.pending.len() >= VAD_CHUNK_SAMPLES {
            let chunk: Vec<_> = self.pending.drain(..VAD_CHUNK_SAMPLES).collect();
            if matches!(
                self.segmenter.process(self.vad.process_chunk(&chunk)?),
                VadEvent::SpeechEnd
            ) {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

pub(super) fn enqueue_segments(
    segmenter: &mut Segmenter,
    samples: &[f32],
    tx: &Sender<Collected>,
    dropped: &mut u64,
) {
    for segment in segmenter.push(samples) {
        enqueue_segment(segment, tx, dropped);
    }
}

pub(super) fn enqueue_segment(segment: Segment, tx: &Sender<Collected>, dropped: &mut u64) {
    let count = segment.owned_end - segment.owned_start;
    if tx.try_send(Collected::Segment(segment)).is_err() {
        *dropped += count as u64;
    }
}

/// Offers the newest uncommitted audio for an interim decode when the cadence
/// allows. Previews travel on their own single-slot channel: a slow worker
/// simply misses a tick, and a preview can never displace a real segment or
/// register as dropped audio.
pub(super) fn offer_preview(
    cadence: &mut PreviewCadence,
    segmenter: &Segmenter,
    new_samples: usize,
    previews: Option<&Sender<Preview>>,
) -> bool {
    let Some(previews) = previews else {
        return false;
    };
    if !cadence.observe(new_samples, segmenter.owned_pending()) {
        return false;
    }
    match segmenter.preview(PREVIEW_MAX_OWNED_SAMPLES) {
        Some(preview) => previews.try_send(preview).is_ok(),
        None => false,
    }
}

/// Opens the microphone and records `first`. When a session ends with
/// `keep_warm`, the same capture stream stays open and the audio heard until
/// the next `start` becomes the start of that session, so nothing said between
/// two sessions is lost.
pub(super) fn collect(device: Option<String>, first: Attach, warm: WarmSlot) {
    let capture = match Capture::start(&device) {
        Ok(capture) => capture,
        Err(error) => {
            fail(first, format!("Could not start audio capture: {error:#}"));
            return;
        }
    };
    let mut attach = first;
    let mut preroll = Vec::new();
    loop {
        if !record(&capture, attach, std::mem::take(&mut preroll)) {
            return;
        }
        match wait_warm(&capture, &warm) {
            Some((next, buffered)) => {
                attach = next;
                preroll = buffered;
            }
            None => return,
        }
    }
}

fn fail(attach: Attach, message: String) {
    let _ = attach.started.send(Err(message.clone()));
    let _ = attach.tx.send(Collected::Done {
        dropped: 0,
        error: Some(message),
    });
}

/// Records one session. Returns true when the microphone should stay warm for
/// the next one; otherwise the capture has been stopped.
fn record(capture: &Capture, attach: Attach, preroll: Vec<f32>) -> bool {
    let Attach {
        job_id,
        control,
        keep_warm,
        tx,
        previews,
        started,
        mut capture_end,
    } = attach;
    protocol::emit(protocol::session_started(&job_id));
    let _ = started.send(Ok(()));
    let mut resampler = StreamingResampler::new(capture.sample_rate, 16000);
    let mut segmenter = Segmenter::default();
    let mut cadence = PreviewCadence::default();
    let mut dropped = 0;
    let dropped_before = capture.stats.dropped.load(Ordering::Relaxed);
    let mut error = None;
    let capture_start = std::time::Instant::now();
    if !preroll.is_empty() {
        let resampled = resampler.process(&preroll);
        enqueue_segments(&mut segmenter, &resampled, &tx, &mut dropped);
        offer_preview(&mut cadence, &segmenter, resampled.len(), previews.as_ref());
    }
    while control.load(Ordering::SeqCst) == RUN {
        match capture.receiver.recv_timeout(Duration::from_millis(10)) {
            Ok(samples) => {
                let resampled = resampler.process(&samples);
                enqueue_segments(&mut segmenter, &resampled, &tx, &mut dropped);
                offer_preview(&mut cadence, &segmenter, resampled.len(), previews.as_ref());
                if let Some(detector) = capture_end.as_mut() {
                    match detector.process(&resampled) {
                        Ok(true) => break,
                        Ok(false) => {}
                        Err(vad_error) => {
                            error = Some(format!("Capture VAD failed: {vad_error}"));
                            break;
                        }
                    }
                    if capture_start.elapsed() >= Duration::from_secs(30) {
                        break;
                    }
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(_) => {
                error = Some("Audio capture disconnected".into());
                break;
            }
        }
        if capture.stats.errors.load(Ordering::Relaxed) > 0 {
            error = Some("The audio device reported a capture error".into());
            break;
        }
    }
    // No previews once the user has stopped: the worker should spend its time
    // on the final transcript, and closing the channel tells it so.
    drop(previews);
    let stay_warm = error.is_none()
        && capture_end.is_none()
        && control.load(Ordering::SeqCst) == super::STOP
        && keep_warm.load(Ordering::SeqCst);
    if !stay_warm {
        if let Err(stop_error) = capture.stop() {
            error = Some(format!("Could not stop capture cleanly: {stop_error}"));
        }
    }
    protocol::emit(protocol::transcribing(&job_id));
    // Audio already queued was heard before `stop`, so it belongs to this
    // session; anything arriving after this point goes to the next one.
    finish_audio(
        &capture.receiver,
        &mut resampler,
        &mut segmenter,
        &tx,
        &mut dropped,
    );
    dropped += ((capture.stats.dropped.load(Ordering::Relaxed) - dropped_before) * 16000)
        .div_ceil(capture.sample_rate as u64);
    let _ = tx.send(Collected::Done { dropped, error });
    stay_warm
}

/// Keeps listening between sessions, buffering the newest audio, until the
/// next session attaches or `WARM_TIMEOUT` passes.
fn wait_warm(capture: &Capture, warm: &WarmSlot) -> Option<(Attach, Vec<f32>)> {
    let (attach_tx, attach_rx) = crossbeam_channel::bounded(1);
    if let Ok(mut slot) = warm.lock() {
        *slot = Some(attach_tx);
    } else {
        let _ = capture.stop();
        return None;
    }
    let mut ring = PrerollRing::new(capture.sample_rate as usize * WARM_PREROLL_SECONDS);
    let deadline = Instant::now() + WARM_TIMEOUT;
    loop {
        if let Ok(attach) = attach_rx.try_recv() {
            return Some((attach, ring.into_vec()));
        }
        let failed = match capture.receiver.recv_timeout(Duration::from_millis(10)) {
            Ok(samples) => {
                ring.push(&samples);
                capture.stats.errors.load(Ordering::Relaxed) > 0
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => false,
            Err(_) => true,
        };
        if failed || Instant::now() >= deadline {
            let taken = warm
                .lock()
                .map(|mut slot| slot.take().is_none())
                .unwrap_or(true);
            let _ = capture.stop();
            if taken {
                // A `start` already took the sender; answer it rather than
                // leaving it waiting for a thread that is gone.
                if let Ok(attach) = attach_rx.recv_timeout(Duration::from_secs(1)) {
                    fail(
                        attach,
                        "The warm microphone closed before the session started".into(),
                    );
                }
            }
            return None;
        }
    }
}

pub(super) fn finish_audio(
    receiver: &Receiver<Vec<f32>>,
    resampler: &mut StreamingResampler,
    segmenter: &mut Segmenter,
    tx: &Sender<Collected>,
    dropped: &mut u64,
) {
    for samples in receiver.try_iter() {
        enqueue_segments(segmenter, &resampler.process(&samples), tx, dropped);
    }
    enqueue_segments(segmenter, &resampler.finish(), tx, dropped);
    if let Some(segment) = segmenter.finish() {
        enqueue_segment(segment, tx, dropped);
    }
}

#[cfg(test)]
mod tests {
    use super::PrerollRing;

    #[test]
    fn preroll_keeps_only_the_newest_samples() {
        let mut ring = PrerollRing::new(4);
        ring.push(&[1.0, 2.0]);
        ring.push(&[3.0, 4.0, 5.0]);
        assert_eq!(ring.into_vec(), vec![2.0, 3.0, 4.0, 5.0]);

        let mut short = PrerollRing::new(10);
        short.push(&[1.0]);
        assert_eq!(short.into_vec(), vec![1.0]);
    }
}
