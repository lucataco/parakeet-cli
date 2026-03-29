/// Benchmarks for mel spectrogram computation.
///
/// Measures the CPU-bound FFT + mel filterbank pipeline across
/// various audio durations. This is the first stage after audio
/// loading and is entirely CPU-bound (no ONNX inference).
mod common;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parakeet_cli::audio::{MelConfig, compute_mel_spectrogram};

fn bench_mel_spectrogram(c: &mut Criterion) {
    let config = MelConfig::default();
    let durations: &[(f32, &str)] = &[
        (1.0, "1s"),
        (5.0, "5s"),
        (10.0, "10s"),
        (30.0, "30s"),
        (60.0, "60s"),
    ];

    let mut group = c.benchmark_group("mel_spectrogram");

    for &(duration, label) in durations {
        let audio = common::generate_sine_audio(duration, 440.0);
        let n_samples = audio.len() as u64;

        group.throughput(Throughput::Elements(n_samples));
        group.bench_with_input(BenchmarkId::new("sine", label), &audio, |b, samples| {
            b.iter(|| {
                let _features = compute_mel_spectrogram(samples, &config);
            });
        });
    }

    // Also benchmark with noise (different spectral content stresses the
    // filterbank differently, though compute cost is the same).
    let noise_10s = common::generate_noise_audio(10.0);
    group.throughput(Throughput::Elements(noise_10s.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("noise", "10s"),
        &noise_10s,
        |b, samples| {
            b.iter(|| {
                let _features = compute_mel_spectrogram(samples, &config);
            });
        },
    );

    group.finish();
}

fn bench_mel_realtime_factor(c: &mut Criterion) {
    // Single benchmark that prints the realtime factor for reference.
    // Uses 10 seconds of audio as the reference duration.
    let config = MelConfig::default();
    let audio = common::generate_sine_audio(10.0, 440.0);

    c.bench_function("mel_rtf_10s", |b| {
        b.iter(|| {
            let _features = compute_mel_spectrogram(&audio, &config);
        });
    });
}

criterion_group!(benches, bench_mel_spectrogram, bench_mel_realtime_factor);
criterion_main!(benches);
