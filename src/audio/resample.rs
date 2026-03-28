/// Audio resampling to 16kHz mono.
///
/// For Phase 2, we use simple linear interpolation resampling.
/// This is sufficient for file transcription where quality loss
/// from linear interpolation is minimal. Phase 3 may upgrade
/// to a proper sinc-based resampler (rubato) for streaming.
use anyhow::{Context, Result};

/// Stateful streaming linear resampler.
///
/// Preserves source-sample leftovers and fractional position across calls so
/// callback-sized chunks behave like a single continuous stream.
pub struct StreamingResampler {
    src_rate: u32,
    target_rate: u32,
    step: f64,
    input: Vec<f32>,
    position: f64,
    total_input_samples: usize,
    total_output_samples: usize,
}

impl StreamingResampler {
    pub fn new(src_rate: u32, target_rate: u32) -> Self {
        Self {
            src_rate,
            target_rate,
            step: src_rate as f64 / target_rate as f64,
            input: Vec::new(),
            position: 0.0,
            total_input_samples: 0,
            total_output_samples: 0,
        }
    }

    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        if self.src_rate == self.target_rate {
            self.total_input_samples += samples.len();
            self.total_output_samples += samples.len();
            return samples.to_vec();
        }

        self.input.extend_from_slice(samples);
        self.total_input_samples += samples.len();

        let mut output = Vec::new();
        while self.position + 1.0 < self.input.len() as f64 {
            output.push(self.sample_at(self.position));
            self.position += self.step;
        }

        self.compact_input();
        self.total_output_samples += output.len();
        output
    }

    pub fn finish(&mut self) -> Vec<f32> {
        if self.src_rate == self.target_rate {
            return Vec::new();
        }

        if self.input.is_empty() {
            return Vec::new();
        }

        let target_total = (self.total_input_samples as f64 * self.target_rate as f64
            / self.src_rate as f64)
            .ceil() as usize;

        let mut output = Vec::new();
        while self.total_output_samples + output.len() < target_total {
            output.push(self.sample_at_clamped(self.position));
            self.position += self.step;
        }

        self.total_output_samples += output.len();
        self.input.clear();
        self.position = 0.0;
        output
    }

    fn sample_at(&self, pos: f64) -> f32 {
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        self.input[idx] * (1.0 - frac) + self.input[idx + 1] * frac
    }

    fn sample_at_clamped(&self, pos: f64) -> f32 {
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let a = self.input.get(idx).copied().unwrap_or(0.0);
        let b = self.input.get(idx + 1).copied().unwrap_or(a);
        a * (1.0 - frac) + b * frac
    }

    fn compact_input(&mut self) {
        if self.input.len() <= 1 {
            return;
        }

        let consumed = (self.position.floor() as usize).min(self.input.len() - 1);
        if consumed > 0 {
            self.input.drain(..consumed);
            self.position -= consumed as f64;
        }
    }
}

/// Resample audio from source sample rate to target sample rate using linear interpolation.
///
/// For 16kHz -> 16kHz this is a no-op (returns a clone).
pub fn resample_linear(samples: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
    if src_rate == target_rate {
        return samples.to_vec();
    }

    let ratio = target_rate as f64 / src_rate as f64;
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos.floor() as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        let sample = if src_idx + 1 < samples.len() {
            samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac
        } else if src_idx < samples.len() {
            samples[src_idx]
        } else {
            0.0
        };
        output.push(sample);
    }

    output
}

/// Convert stereo interleaved samples to mono by averaging channels.
pub fn stereo_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }

    let ch = channels as usize;
    let n_frames = samples.len() / ch;
    let mut mono = Vec::with_capacity(n_frames);

    for i in 0..n_frames {
        let mut sum = 0.0f32;
        for c in 0..ch {
            sum += samples[i * ch + c];
        }
        mono.push(sum / ch as f32);
    }

    mono
}

/// Load a WAV file and return mono 16kHz f32 samples.
pub fn load_wav_file(path: &std::path::Path, verbose: bool) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    let channels = spec.channels;
    let sample_rate = spec.sample_rate;

    if verbose {
        println!(
            "Audio: {}ch, {}Hz, {:?} {}bit",
            channels, sample_rate, spec.sample_format, spec.bits_per_sample
        );
    }

    // Read all samples as f32
    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
    };

    // Convert to mono
    let mono = stereo_to_mono(&raw_samples, channels);

    // Resample to 16kHz
    let resampled = resample_linear(&mono, sample_rate, 16000);

    if verbose {
        println!(
            "Loaded {} samples ({:.2}s at 16kHz)",
            resampled.len(),
            resampled.len() as f64 / 16000.0
        );
    }

    Ok(resampled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_same_rate() {
        let input = vec![1.0, 2.0, 3.0];
        let output = resample_linear(&input, 16000, 16000);
        assert_eq!(output, input);
    }

    #[test]
    fn test_resample_upsample() {
        let input = vec![0.0, 1.0];
        let output = resample_linear(&input, 8000, 16000);
        // 2x upsample: should produce ~4 samples with interpolation
        assert!(output.len() >= 3);
        // First sample should be 0
        assert!((output[0]).abs() < 1e-6);
    }

    #[test]
    fn test_stereo_to_mono() {
        let stereo = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0];
        let mono = stereo_to_mono(&stereo, 2);
        assert_eq!(mono.len(), 3);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
        assert!((mono[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_streaming_resampler_matches_one_shot() {
        let input: Vec<f32> = (0..4800)
            .map(|i| ((i as f32) / 19.0).sin() * 0.5 + ((i as f32) / 7.0).cos() * 0.1)
            .collect();

        let expected = resample_linear(&input, 48_000, 16_000);

        let mut streaming = StreamingResampler::new(48_000, 16_000);
        let mut actual = Vec::new();
        for chunk in input.chunks(137) {
            actual.extend(streaming.process(chunk));
        }
        actual.extend(streaming.finish());

        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4, "streaming={a} expected={b}");
        }
    }
}
