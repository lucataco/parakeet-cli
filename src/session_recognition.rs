use crate::segments::{FRAME_SAMPLES, Segment};
use crate::{
    audio,
    model::ParakeetModel,
    vad::{SileroVad, VAD_CHUNK_SAMPLES},
};
use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;
use std::time::Instant;

pub struct Recognizer {
    model: ParakeetModel,
    vad: SileroVad,
}

/// Seconds spent in each recognition stage, summed over a session's segments.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct StageTimes {
    pub vad: f64,
    pub mel: f64,
    pub encoder: f64,
    pub decoder: f64,
}

#[derive(Default)]
pub struct SessionResult {
    pub tokens: Vec<usize>,
    pub text: String,
    pub failed_segments: u64,
    pub dropped_samples: u64,
    pub message: Option<String>,
    pub duration: f64,
    pub inference_time: f64,
    pub stages: StageTimes,
}

impl SessionResult {
    pub fn completion(&self, session_id: &str) -> Value {
        crate::serve::protocol::completion(self, session_id)
    }
}

impl Recognizer {
    pub fn load(model_dir: &Path, vad_path: &Path, coreml: bool) -> Result<Self> {
        let config: Value = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
        anyhow::ensure!(
            config["subsampling_factor"].as_u64() == Some(8),
            "Unsupported encoder frame stride"
        );
        Ok(Self {
            model: ParakeetModel::load(model_dir, coreml, false)?,
            vad: SileroVad::load(vad_path, false)?,
        })
    }

    pub fn append(&mut self, segment: Segment, result: &mut SessionResult) {
        result.duration += (segment.owned_end - segment.owned_start) as f64 / 16000.0;
        let started = Instant::now();
        let stages = &mut result.stages;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.transcribe(segment, stages)
        }))
        .unwrap_or_else(|_| {
            Err(anyhow::anyhow!(
                "Recognition panicked while processing a segment"
            ))
        });
        match outcome {
            Ok(tokens) => result.tokens.extend(tokens),
            Err(error) => {
                result.failed_segments += 1;
                result.message = Some(format!("A segment failed: {error:#}"));
                eprintln!("Segment failed: {error:#}");
            }
        }
        result.inference_time += started.elapsed().as_secs_f64();
    }

    /// Recognizes uncommitted audio for an interim preview. Failures and panics
    /// are swallowed: a preview is advisory and never counts toward the
    /// session's loss accounting or its final transcript.
    pub fn preview(&mut self, segment: Segment) -> Option<Vec<usize>> {
        let mut discarded = StageTimes::default();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.transcribe(segment, &mut discarded)
        }))
        .ok()
        .and_then(|outcome| outcome.ok())
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        self.model.tokenizer.decode(tokens)
    }

    fn transcribe(&mut self, segment: Segment, stages: &mut StageTimes) -> Result<Vec<usize>> {
        let started = Instant::now();
        self.vad.reset();
        let mut speech = false;
        for chunk in segment.samples.chunks(VAD_CHUNK_SAMPLES) {
            let mut padded = [0.0; VAD_CHUNK_SAMPLES];
            padded[..chunk.len()].copy_from_slice(chunk);
            speech |= self.vad.process_chunk(&padded)? >= 0.5;
        }
        stages.vad += started.elapsed().as_secs_f64();
        if !speech {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let features =
            audio::compute_mel_spectrogram(&segment.samples, &audio::MelConfig::default());
        stages.mel += started.elapsed().as_secs_f64();
        let started = Instant::now();
        let (encoded, shape, lengths) = self.model.encoder.encode(&features)?;
        stages.encoder += started.elapsed().as_secs_f64();
        anyhow::ensure!(shape.len() == 3, "Invalid encoder shape");
        let valid = usize::try_from(*lengths.first().context("No encoder length")?)?.min(shape[2]);
        let start = (segment.owned_start / FRAME_SAMPLES).min(valid);
        let end = segment.owned_end.div_ceil(FRAME_SAMPLES).min(valid);
        let frames = end.saturating_sub(start);
        anyhow::ensure!(frames > 0, "Encoder returned no frames for recorded speech");
        let started = Instant::now();
        let tokens = self.model.decoder.decode_greedy_window(
            &encoded,
            &shape,
            valid as i64,
            self.model.tokenizer.blank_id,
            start..end,
        );
        stages.decoder += started.elapsed().as_secs_f64();
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_capture_with_drops_is_failure_not_silence() {
        let result = SessionResult {
            dropped_samples: 12,
            ..Default::default()
        };
        assert_eq!(result.completion("test")["status"], "error");
    }
    #[test]
    fn partial_and_multiline_are_explicit() {
        let result = SessionResult {
            text: "first\nsecond 🦜".into(),
            failed_segments: 1,
            ..Default::default()
        };
        let wire = result.completion("test").to_string();
        assert!(!wire.contains('\n'));
        let value: Value = serde_json::from_str(&wire).unwrap();
        assert_eq!(value["status"], "partial");
        assert_eq!(value["text"], result.text);
    }
}
