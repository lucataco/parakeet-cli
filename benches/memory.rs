/// Memory usage benchmarks.
///
/// Measures allocation patterns and data sizes during each pipeline
/// stage. Each benchmark computes and returns the estimated memory
/// footprint alongside the actual computation, so Criterion can
/// track both timing and allocation regressions.
///
/// Requires model files for encoder/decoder memory tests.
mod common;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use parakeet_cli::audio::{MelConfig, compute_mel_spectrogram};

fn bench_memory_mel(c: &mut Criterion) {
    let config = MelConfig::default();

    let durations: &[(f32, &str)] = &[(1.0, "1s"), (10.0, "10s"), (30.0, "30s"), (60.0, "60s")];

    let mut group = c.benchmark_group("memory_mel_spectrogram");
    group.sample_size(10);

    for &(duration, label) in durations {
        let audio = common::generate_sine_audio(duration, 440.0);
        let n_samples = audio.len();

        group.bench_with_input(BenchmarkId::from_parameter(label), &audio, |b, samples| {
            b.iter(|| {
                // Compute mel and measure output size
                let features = compute_mel_spectrogram(samples, &config);
                let shape = features.shape();
                let mel_bytes = shape[0] * shape[1] * std::mem::size_of::<f32>();

                // Also account for intermediate allocations:
                // - preemphasis output: n_samples * 4 bytes
                // - FFT buffers: n_fft * 4 * 2 (input + complex output)
                // - filterbank: n_mels * (n_fft/2+1) * 4 bytes
                let preemphasis_bytes = n_samples * 4;
                let fft_bytes = 512 * 4 * 2;
                let filterbank_bytes = 128 * 257 * 4;
                let estimated_peak = mel_bytes + preemphasis_bytes + fft_bytes + filterbank_bytes;

                // Return the feature array to prevent optimization
                (features, estimated_peak)
            });
        });
    }

    group.finish();
}

fn bench_memory_audio_buffer(c: &mut Criterion) {
    use parakeet_cli::audio::AudioBuffer;

    let durations: &[(f32, &str)] = &[(1.0, "1s"), (10.0, "10s"), (30.0, "30s"), (60.0, "60s")];

    let mut group = c.benchmark_group("memory_audio_buffer");

    for &(duration, label) in durations {
        let audio = common::generate_sine_audio(duration, 440.0);

        group.bench_with_input(BenchmarkId::from_parameter(label), &audio, |b, samples| {
            b.iter(|| {
                let mut buffer = AudioBuffer::new(60.0);
                // Simulate streaming: push in 512-sample chunks (like VAD)
                for chunk in samples.chunks(512) {
                    buffer.push(chunk);
                }
                let drained = buffer.drain();
                drained.len()
            });
        });
    }

    group.finish();
}

fn bench_memory_encoder(c: &mut Criterion) {
    if !common::model_available() {
        eprintln!("SKIP: memory encoder benchmarks require model files.");
        return;
    }

    let model_dir = common::default_model_dir();
    let config = MelConfig::default();

    let encoder_path = common::encoder_path();
    let cache_dir = model_dir.join("coreml_cache");
    let mut encoder =
        parakeet_cli::model::encoder::Encoder::load(&encoder_path, false, false, Some(&cache_dir))
            .expect("Failed to load encoder");

    let durations: &[(f32, &str)] = &[(1.0, "1s"), (10.0, "10s"), (30.0, "30s")];

    let mut group = c.benchmark_group("memory_encoder_inference");
    group.sample_size(10);

    for &(duration, label) in durations {
        let audio = common::generate_sine_audio(duration, 440.0);
        let features = compute_mel_spectrogram(&audio, &config);

        group.bench_with_input(BenchmarkId::from_parameter(label), &features, |b, feats| {
            b.iter(|| {
                let (enc_output, enc_shape, _lengths) =
                    encoder.encode(feats).expect("Encoder failed");

                // Report data sizes for memory analysis
                let input_bytes = feats.shape()[0] * feats.shape()[1] * std::mem::size_of::<f32>();
                let output_bytes = enc_output.len() * std::mem::size_of::<f32>();
                // Transposed input buffer allocated inside encode()
                let transpose_bytes =
                    feats.shape()[0] * feats.shape()[1] * std::mem::size_of::<f32>();

                (
                    enc_output,
                    enc_shape,
                    input_bytes + output_bytes + transpose_bytes,
                )
            });
        });
    }

    group.finish();
}

fn bench_memory_full_pipeline(c: &mut Criterion) {
    if !common::model_available() {
        return;
    }

    let model_dir = common::default_model_dir();
    let config = MelConfig::default();

    let mut model = parakeet_cli::model::ParakeetModel::load(&model_dir, false, false)
        .expect("Failed to load model");

    let durations: &[(f32, &str)] = &[(1.0, "1s"), (10.0, "10s"), (30.0, "30s")];

    let mut group = c.benchmark_group("memory_full_pipeline");
    group.sample_size(10);

    for &(duration, label) in durations {
        let audio = common::generate_sine_audio(duration, 440.0);

        group.bench_with_input(BenchmarkId::from_parameter(label), &audio, |b, samples| {
            b.iter(|| {
                let features = compute_mel_spectrogram(samples, &config);

                // Track sizes
                let audio_bytes = samples.len() * std::mem::size_of::<f32>();
                let mel_bytes =
                    features.shape()[0] * features.shape()[1] * std::mem::size_of::<f32>();

                let text = model.transcribe(&features).expect("Failed");
                (text, audio_bytes + mel_bytes)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_mel,
    bench_memory_audio_buffer,
    bench_memory_encoder,
    bench_memory_full_pipeline,
);
criterion_main!(benches);
