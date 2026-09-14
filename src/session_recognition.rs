use crate::segments::{FRAME_SAMPLES, Segment};
use crate::{
    audio,
    model::ParakeetModel,
    vad::{SileroVad, VAD_CHUNK_SAMPLES},
};
use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;

pub struct Recognizer {
    model: ParakeetModel,
    vad: SileroVad,
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
        let started = std::time::Instant::now();
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.transcribe(segment)))
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

    pub fn decode(&self, tokens: &[usize]) -> String {
        self.model.tokenizer.decode(tokens)
    }

    fn transcribe(&mut self, segment: Segment) -> Result<Vec<usize>> {
        self.vad.reset();
        let mut speech = false;
        for chunk in segment.samples.chunks(VAD_CHUNK_SAMPLES) {
            let mut padded = [0.0; VAD_CHUNK_SAMPLES];
            padded[..chunk.len()].copy_from_slice(chunk);
            speech |= self.vad.process_chunk(&padded)? >= 0.5;
        }
        if !speech {
            return Ok(Vec::new());
        }
        let features =
            audio::compute_mel_spectrogram(&segment.samples, &audio::MelConfig::default());
        let (encoded, shape, lengths) = self.model.encoder.encode(&features)?;
        anyhow::ensure!(shape.len() == 3, "Invalid encoder shape");
        let valid = usize::try_from(*lengths.first().context("No encoder length")?)?.min(shape[2]);
        let start = (segment.owned_start / FRAME_SAMPLES).min(valid);
        let end = segment.owned_end.div_ceil(FRAME_SAMPLES).min(valid);
        let frames = end.saturating_sub(start);
        anyhow::ensure!(frames > 0, "Encoder returned no frames for recorded speech");
        self.model.decoder.decode_greedy_window(
            &encoded,
            &shape,
            valid as i64,
            self.model.tokenizer.blank_id,
            start..end,
        )
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
