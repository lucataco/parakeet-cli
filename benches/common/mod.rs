#![allow(dead_code)]
/// Shared helpers for performance benchmarks.
///
/// Provides synthetic audio generation and model path detection
/// so benchmarks are self-contained (no WAV fixture files needed).
use std::path::PathBuf;

/// Generate a mono sine-wave signal at 16 kHz.
///
/// Useful for simulating tonal / speech-like audio without needing
/// real recordings.
pub fn generate_sine_audio(duration_secs: f32, freq_hz: f32) -> Vec<f32> {
    let sample_rate = 16_000.0_f32;
    let n_samples = (duration_secs * sample_rate) as usize;
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.5
        })
        .collect()
}

/// Generate white noise at 16 kHz using a simple LCG PRNG.
///
/// Deterministic (seeded) so benchmarks are reproducible.
pub fn generate_noise_audio(duration_secs: f32) -> Vec<f32> {
    let sample_rate = 16_000.0_f32;
    let n_samples = (duration_secs * sample_rate) as usize;
    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234;
    (0..n_samples)
        .map(|_| {
            // Simple LCG
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-0.5, 0.5]
            ((rng_state >> 33) as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

/// Generate stereo interleaved audio at a given sample rate.
///
/// Returns interleaved L/R samples (total length = n_samples * 2).
pub fn generate_stereo_audio(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let n_samples = (duration_secs * sample_rate as f32) as usize;
    let mut out = Vec::with_capacity(n_samples * 2);
    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;
        let left = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.4;
        let right = (2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.3;
        out.push(left);
        out.push(right);
    }
    out
}

/// Return the default model directory path (same logic as the CLI).
/// Checks v3 first, then falls back to v2 for backward compatibility.
pub fn default_model_dir() -> PathBuf {
    let base = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("parakeet")
        .join("models");

    let v3 = base.join("parakeet-tdt-0.6b-v3");
    if v3.exists() {
        return v3;
    }
    let v2 = base.join("parakeet-tdt-0.6b-v2");
    if v2.exists() {
        return v2;
    }
    // Default to v3 path even if it doesn't exist
    v3
}

/// Check whether any Parakeet encoder model is available on disk.
pub fn model_available() -> bool {
    let dir = default_model_dir();
    dir.join("encoder-model.fp16.onnx").exists()
        || dir.join("encoder-model.int8.onnx").exists()
        || dir.join("encoder-model.onnx").exists()
}

/// Find the encoder model path (FP16 > INT8 > FP32).
pub fn encoder_path() -> PathBuf {
    let dir = default_model_dir();
    if dir.join("encoder-model.fp16.onnx").exists() {
        dir.join("encoder-model.fp16.onnx")
    } else if dir.join("encoder-model.int8.onnx").exists() {
        dir.join("encoder-model.int8.onnx")
    } else {
        dir.join("encoder-model.onnx")
    }
}

/// Find the decoder model path (FP16 > INT8 > FP32).
pub fn decoder_path() -> PathBuf {
    let dir = default_model_dir();
    if dir.join("decoder_joint-model.fp16.onnx").exists() {
        dir.join("decoder_joint-model.fp16.onnx")
    } else if dir.join("decoder_joint-model.int8.onnx").exists() {
        dir.join("decoder_joint-model.int8.onnx")
    } else {
        dir.join("decoder_joint-model.onnx")
    }
}

/// Check whether the Silero VAD model is available on disk.
pub fn vad_model_available() -> bool {
    let dir = default_model_dir();
    dir.join("silero_vad.onnx").exists()
}
