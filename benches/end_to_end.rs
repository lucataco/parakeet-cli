/// End-to-end benchmarks for the full transcription pipeline.
///
/// Measures the complete path: audio -> mel spectrogram -> encoder ->
/// decoder -> tokenizer -> text. This is what a user experiences
/// when running `parakeet transcribe`.
///
/// Also includes a batch throughput test that transcribes multiple
/// files sequentially to measure sustained performance.
///
/// Requires model files to be downloaded (`parakeet download`).
mod common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use parakeet_cli::audio::{compute_mel_spectrogram, MelConfig};
use parakeet_cli::model::ParakeetModel;

fn bench_e2e_transcription(c: &mut Criterion) {
    if !common::model_available() {
        eprintln!(
            "SKIP: end-to-end benchmarks require model files. Run `parakeet download` first."
        );
        return;
    }

    let model_dir = common::default_model_dir();

    // Load model once
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
                // Full pipeline: mel -> encode -> decode -> text
                let features = compute_mel_spectrogram(samples, &config);
                let text = model.transcribe(&features).expect("Transcription failed");
                text
            });
        });
    }

    group.finish();
}

fn bench_e2e_with_mel_precomputed(c: &mut Criterion) {
    // Measures only the model inference portion (encoder + decoder + tokenizer)
    // with mel spectrogram pre-computed, to isolate CPU vs ML costs.
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
        b.iter(|| {
            let text = model.transcribe(&features).expect("Transcription failed");
            text
        });
    });

    group.finish();
}

fn bench_e2e_batch_throughput(c: &mut Criterion) {
    // Simulate transcribing multiple utterances back-to-back.
    // This measures sustained throughput, including any overhead
    // from decoder state resets between utterances.
    if !common::model_available() {
        return;
    }

    let model_dir = common::default_model_dir();
    let mut model =
        ParakeetModel::load(&model_dir, true, false).expect("Failed to load Parakeet model");

    let config = MelConfig::default();

    // Generate 10 different 5-second "utterances"
    let utterances: Vec<Vec<f32>> = (0..10)
        .map(|i| {
            let freq = 200.0 + (i as f32) * 50.0; // Vary frequency slightly
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
    // Explicit RTF measurement: reports time to transcribe N seconds of audio.
    // The RTF = audio_duration / wall_clock_time.
    // Values > 1.0 mean faster than realtime.
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
