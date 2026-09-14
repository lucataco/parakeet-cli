mod common;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use parakeet_cli::audio::{MelConfig, compute_mel_spectrogram};
use parakeet_cli::model::ParakeetModel;

fn bench_e2e_transcription(c: &mut Criterion) {
    if !common::model_available() {
        eprintln!(
            "SKIP: end-to-end benchmarks require model files. Run `parakeet download` first."
        );
        return;
    }

    let model_dir = common::default_model_dir();

    let mut model =
        ParakeetModel::load(&model_dir, true, false).expect("Failed to load Parakeet model");

    let config = MelConfig::default();

    let durations: &[(f32, &str)] = &[(1.0, "1s"), (5.0, "5s"), (10.0, "10s"), (30.0, "30s")];

    let mut group = c.benchmark_group("e2e_transcription");
    group.sample_size(10);

    for &(duration, label) in durations {
        let audio = common::generate_sine_audio(duration, 440.0);

        group.bench_with_input(BenchmarkId::from_parameter(label), &audio, |b, samples| {
            b.iter(|| {
                let features = compute_mel_spectrogram(samples, &config);
                model.transcribe(&features).expect("Transcription failed")
            });
        });
    }

    group.finish();
}

fn bench_e2e_with_mel_precomputed(c: &mut Criterion) {
    if !common::model_available() {
        return;
    }

    let model_dir = common::default_model_dir();
    let mut model =
        ParakeetModel::load(&model_dir, true, false).expect("Failed to load Parakeet model");

    let config = MelConfig::default();
    let audio = common::generate_sine_audio(10.0, 440.0);
    let features = compute_mel_spectrogram(&audio, &config);

    let mut group = c.benchmark_group("e2e_inference_only");
    group.sample_size(10);

    group.bench_function("10s_precomputed_mel", |b| {
        b.iter(|| model.transcribe(&features).expect("Transcription failed"));
    });

    group.finish();
}

fn bench_e2e_batch_throughput(c: &mut Criterion) {
    if !common::model_available() {
        return;
    }

    let model_dir = common::default_model_dir();
    let mut model =
        ParakeetModel::load(&model_dir, true, false).expect("Failed to load Parakeet model");

    let config = MelConfig::default();

    let utterances: Vec<Vec<f32>> = (0..10)
        .map(|i| {
            let freq = 200.0 + (i as f32) * 50.0;
            common::generate_sine_audio(5.0, freq)
        })
        .collect();

    let mut group = c.benchmark_group("e2e_batch_throughput");
    group.sample_size(10);

    group.bench_function("10x5s_sequential", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(10);
            for audio in &utterances {
                let features = compute_mel_spectrogram(audio, &config);
                let text = model.transcribe(&features).expect("Transcription failed");
                results.push(text);
            }
            results
        });
    });

    group.finish();
}

fn bench_e2e_realtime_factor(c: &mut Criterion) {
    if !common::model_available() {
        return;
    }

    let model_dir = common::default_model_dir();
    let mut model =
        ParakeetModel::load(&model_dir, true, false).expect("Failed to load Parakeet model");

    let config = MelConfig::default();

    let durations: &[(f32, &str)] = &[(5.0, "5s"), (10.0, "10s"), (30.0, "30s")];

    let mut group = c.benchmark_group("e2e_rtf");
    group.sample_size(10);

    for &(duration, label) in durations {
        let audio = common::generate_sine_audio(duration, 440.0);

        group.bench_with_input(BenchmarkId::from_parameter(label), &audio, |b, samples| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let start = std::time::Instant::now();
                    let features = compute_mel_spectrogram(samples, &config);
                    let _text = model.transcribe(&features).expect("Transcription failed");
                    total += start.elapsed();
                }
                total
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_e2e_transcription,
    bench_e2e_with_mel_precomputed,
    bench_e2e_batch_throughput,
    bench_e2e_realtime_factor,
);
criterion_main!(benches);
