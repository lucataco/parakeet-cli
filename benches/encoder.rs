/// Benchmarks for the ONNX encoder inference.
///
/// This is the most expensive stage of the pipeline.
/// Benchmarks measure wall-clock time and realtime factor for
/// various audio durations.
///
/// Requires model files to be downloaded (`parakeet download`).
/// Benchmarks are skipped gracefully if the model is not present.
mod common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use parakeet_cli::audio::{compute_mel_spectrogram, MelConfig};
use parakeet_cli::model::encoder::Encoder;

fn bench_encoder_inference(c: &mut Criterion) {
    if !common::model_available() {
        eprintln!("SKIP: encoder benchmarks require model files. Run `parakeet download` first.");
        return;
    }

    let model_dir = common::default_model_dir();
    let encoder_path = common::encoder_path();
    let cache_dir = model_dir.join("coreml_cache");

    let mut encoder = Encoder::load(&encoder_path, false, false, Some(&cache_dir))
        .expect("Failed to load encoder for benchmarks");

    let config = MelConfig::default();

    let durations: &[(f32, &str)] = &[(1.0, "1s"), (5.0, "5s"), (10.0, "10s"), (30.0, "30s")];

    let mut group = c.benchmark_group("encoder_inference");
    // Encoder inference is expensive; use fewer samples
    group.sample_size(10);

    for &(duration, label) in durations {
        let audio = common::generate_sine_audio(duration, 440.0);
        let features = compute_mel_spectrogram(&audio, &config);

        group.bench_with_input(BenchmarkId::from_parameter(label), &features, |b, feats| {
            b.iter(|| {
                let (_enc_output, _enc_shape, _lengths) =
                    encoder.encode(feats).expect("Encoder inference failed");
            });
        });
    }

    group.finish();
}

fn bench_encoder_throughput(c: &mut Criterion) {
    if !common::model_available() {
        return;
    }

    let model_dir = common::default_model_dir();
    let encoder_path = common::encoder_path();
    let cache_dir = model_dir.join("coreml_cache");

    let mut encoder = Encoder::load(&encoder_path, false, false, Some(&cache_dir))
        .expect("Failed to load encoder for benchmarks");

    let config = MelConfig::default();

    // Measure how many audio-seconds per wall-clock second for 10s chunks
    let audio = common::generate_sine_audio(10.0, 440.0);
    let features = compute_mel_spectrogram(&audio, &config);

    let mut group = c.benchmark_group("encoder_throughput");
    group.sample_size(10);

    group.bench_function("10s_chunks", |b| {
        b.iter(|| {
            let (_enc_output, _enc_shape, _lengths) =
                encoder.encode(&features).expect("Encoder inference failed");
        });
    });

    group.finish();
}

criterion_group!(benches, bench_encoder_inference, bench_encoder_throughput);
criterion_main!(benches);
