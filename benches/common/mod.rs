#![allow(dead_code)]
use std::path::PathBuf;

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

pub fn generate_noise_audio(duration_secs: f32) -> Vec<f32> {
    let sample_rate = 16_000.0_f32;
    let n_samples = (duration_secs * sample_rate) as usize;
    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234;
    (0..n_samples)
        .map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 32) as u32) as f32 / u32::MAX as f32 - 0.5
        })
        .collect()
}

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

pub fn default_model_dir() -> PathBuf {
    let base = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("parakeet")
        .join("models");

    base.join("parakeet-tdt-0.6b-v3")
}

pub fn model_available() -> bool {
    let dir = default_model_dir();
    let has_fp16 = dir.join("encoder-model.fp16.onnx").exists()
        && dir.join("decoder_joint-model.fp16.onnx").exists();
    let has_int8 = dir.join("encoder-model.int8.onnx").exists()
        && dir.join("decoder_joint-model.int8.onnx").exists();
    let has_fp32 =
        dir.join("encoder-model.onnx").exists() && dir.join("decoder_joint-model.onnx").exists();

    (has_fp16 || has_int8 || has_fp32)
        && dir.join("vocab.txt").exists()
        && dir.join("config.json").exists()
}

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

pub fn vad_model_available() -> bool {
    let dir = default_model_dir();
    dir.join("silero_vad.onnx").exists()
}
