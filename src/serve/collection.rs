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
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::Duration,
};

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

pub(super) fn collect(
    device: Option<String>,
    job_id: String,
    control: Arc<AtomicU8>,
    tx: Sender<Collected>,
    previews: Option<Sender<Preview>>,
    started: tokio::sync::oneshot::Sender<std::result::Result<(), String>>,
    mut capture_end: Option<CaptureEnd>,
) {
    let capture = match Capture::start(&device) {
        Ok(capture) => capture,
        Err(error) => {
            let message = format!("Could not start audio capture: {error:#}");
            let _ = started.send(Err(message.clone()));
            let _ = tx.send(Collected::Done {
                dropped: 0,
                error: Some(message),
            });
            return;
        }
    };
    protocol::emit(protocol::session_started(&job_id));
    let _ = started.send(Ok(()));
    let mut resampler = StreamingResampler::new(capture.sample_rate, 16000);
    let mut segmenter = Segmenter::default();
    let mut cadence = PreviewCadence::default();
    let mut dropped = 0;
    let mut error = None;
    let capture_start = std::time::Instant::now();
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
    if let Err(stop_error) = capture.stop() {
        error = Some(format!("Could not stop capture cleanly: {stop_error}"));
    }
    protocol::emit(protocol::transcribing(&job_id));
    finish_audio(
        &capture.receiver,
        &mut resampler,
        &mut segmenter,
        &tx,
        &mut dropped,
    );
    dropped += (capture.stats.dropped.load(Ordering::Relaxed) * 16000)
        .div_ceil(capture.sample_rate as u64);
    let _ = tx.send(Collected::Done { dropped, error });
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
