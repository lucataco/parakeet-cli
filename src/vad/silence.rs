use super::{SileroVad, VAD_CHUNK_SAMPLES};
use anyhow::Result;
use std::path::Path;

const MIN_SPEECH_PROBABILITY: f32 = 0.5;

type Classifier = Box<dyn FnMut(&[f32]) -> Result<f32> + Send>;

/// Measures how long a take has been silent, one 512-sample chunk at a time as
/// audio arrives, so the cost per tick never grows with the take.
pub struct SilenceMeter {
    speech_probability: Classifier,
    pending: Vec<f32>,
    classified: usize,
    speech_end: usize,
}

impl SilenceMeter {
    pub fn new(speech_probability: impl FnMut(&[f32]) -> Result<f32> + Send + 'static) -> Self {
        Self {
            speech_probability: Box::new(speech_probability),
            pending: Vec::new(),
            classified: 0,
            speech_end: 0,
        }
    }

    /// A meter with its own Silero instance.
    pub fn silero(path: &Path) -> Result<Self> {
        let mut vad = SileroVad::load(path, false)?;
        Ok(Self::new(move |chunk| vad.process_chunk(chunk)))
    }

    /// After an error the counters no longer describe the audio, so the meter
    /// must be discarded.
    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        self.pending.extend_from_slice(samples);
        while self.pending.len() >= VAD_CHUNK_SAMPLES {
            let speech = (self.speech_probability)(&self.pending[..VAD_CHUNK_SAMPLES])?
                >= MIN_SPEECH_PROBABILITY;
            self.pending.drain(..VAD_CHUNK_SAMPLES);
            self.classified += VAD_CHUNK_SAMPLES;
            if speech {
                self.speech_end = self.classified;
            }
        }
        Ok(())
    }

    /// 16 kHz samples since the end of the last speech chunk: zero right after
    /// speech, and all the audio classified so far before any. Audio short of a
    /// whole chunk is not counted until its chunk completes.
    pub fn trailing_samples(&self) -> usize {
        self.classified - self.speech_end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::protocol::samples_to_ms as ms;

    const CHUNK_MS: u64 = 32;

    fn loud_is_speech() -> SilenceMeter {
        SilenceMeter::new(|chunk| {
            Ok(if chunk.iter().any(|sample| sample.abs() >= 0.1) {
                0.9
            } else {
                0.1
            })
        })
    }

    #[test]
    fn silence_grows_after_speech_and_resets_on_the_next_word() {
        let mut meter = loud_is_speech();
        meter.push(&[0.3; VAD_CHUNK_SAMPLES * 10]).unwrap();
        assert_eq!(meter.trailing_samples(), 0, "the latest chunk is speech");
        meter.push(&[0.0; VAD_CHUNK_SAMPLES * 30]).unwrap();
        assert_eq!(ms(meter.trailing_samples()), 30 * CHUNK_MS);
        meter.push(&[0.3; VAD_CHUNK_SAMPLES]).unwrap();
        assert_eq!(meter.trailing_samples(), 0);
    }

    #[test]
    fn before_any_speech_silence_is_all_the_audio_heard() {
        let mut meter = loud_is_speech();
        assert_eq!(meter.trailing_samples(), 0);
        meter.push(&[0.0; VAD_CHUNK_SAMPLES * 25]).unwrap();
        assert_eq!(meter.trailing_samples(), VAD_CHUNK_SAMPLES * 25);
    }

    #[test]
    fn a_pause_shorter_than_the_endpoint_never_reports_it() {
        let endpoint_ms = 900;
        let straddling_and_filling_chunks_ms = 2 * CHUNK_MS;
        let mut meter = loud_is_speech();
        meter.push(&[0.3; 16_000]).unwrap();
        let mut longest = 0;
        for piece in [0.0; 742 * 16].chunks(480) {
            meter.push(piece).unwrap();
            longest = longest.max(ms(meter.trailing_samples()));
        }
        assert!(
            (742 - straddling_and_filling_chunks_ms..endpoint_ms).contains(&longest),
            "a 742 ms pause measures {longest} ms"
        );
        meter.push(&[0.3; 1_600]).unwrap();
        assert_eq!(meter.trailing_samples(), 0, "speech resumed");
    }

    #[test]
    fn uneven_pushes_classify_the_same_chunks_as_whole_ones() {
        let mut audio = vec![0.3; 10_000];
        audio.extend([0.0; 4_000]);
        audio.extend([0.3; 1_000]);
        audio.extend([0.0; 5_480]);
        let mut whole = loud_is_speech();
        whole.push(&audio).unwrap();
        let mut uneven = loud_is_speech();
        for piece in audio.chunks(480) {
            uneven.push(piece).unwrap();
        }
        assert_eq!(uneven.trailing_samples(), whole.trailing_samples());
        assert_eq!(
            whole.trailing_samples(),
            20_480 - 15_360,
            "the last loud sample (14 999) falls in the chunk ending at 15 360"
        );
    }

    #[test]
    fn a_chunk_at_exactly_the_threshold_is_speech() {
        let mut meter = SilenceMeter::new(|_| Ok(MIN_SPEECH_PROBABILITY));
        meter.push(&[0.0; VAD_CHUNK_SAMPLES * 3]).unwrap();
        assert_eq!(meter.trailing_samples(), 0);
    }

    #[test]
    fn a_classifier_failure_surfaces_instead_of_counting_as_silence() {
        let mut meter = SilenceMeter::new(|_| anyhow::bail!("model failed"));
        assert!(meter.push(&[0.3; VAD_CHUNK_SAMPLES]).is_err());
    }

    #[test]
    fn a_failed_chunk_is_not_counted() {
        let mut calls = 0;
        let mut meter = SilenceMeter::new(move |_| {
            calls += 1;
            if calls == 2 {
                anyhow::bail!("model failed")
            }
            Ok(0.0)
        });
        assert!(meter.push(&[0.0; VAD_CHUNK_SAMPLES * 2]).is_err());
        assert_eq!(meter.trailing_samples(), VAD_CHUNK_SAMPLES);
    }

    #[test]
    fn a_partial_chunk_waits_for_the_rest_of_its_samples() {
        let mut meter = loud_is_speech();
        meter.push(&[0.3; VAD_CHUNK_SAMPLES]).unwrap();
        meter.push(&[0.0; VAD_CHUNK_SAMPLES - 1]).unwrap();
        assert_eq!(meter.trailing_samples(), 0);
        meter.push(&[0.0; 1]).unwrap();
        assert_eq!(meter.trailing_samples(), VAD_CHUNK_SAMPLES);
    }
}
