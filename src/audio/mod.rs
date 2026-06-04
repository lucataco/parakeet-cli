use std::collections::VecDeque;

pub mod buffer;
pub mod capture;
pub mod mel;
pub mod resample;

/// Internal target sample rate for Parakeet audio features.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;
/// Minimum utterance length worth sending to the model (100ms at 16kHz).
pub const MIN_UTTERANCE_SAMPLES: usize = TARGET_SAMPLE_RATE as usize / 10;
/// Audio kept before VAD speech-start so the onset is not clipped (200ms).
pub const PREROLL_SAMPLES: usize = TARGET_SAMPLE_RATE as usize / 5;

pub use buffer::AudioBuffer;
pub use capture::{print_input_devices, start_capture};
pub use mel::{MelConfig, compute_mel_spectrogram};
pub use resample::{StreamingResampler, load_wav_file};

pub fn push_preroll(preroll: &mut VecDeque<f32>, chunk: &[f32], max_samples: usize) {
    preroll.extend(chunk.iter().copied());
    while preroll.len() > max_samples {
        preroll.pop_front();
    }
}
