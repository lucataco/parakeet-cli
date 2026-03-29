/// Benchmarks for audio resampling and format conversion.
///
/// Tests both one-shot and streaming resampling paths, plus
/// stereo-to-mono conversion. These run on the CPU with no
/// model dependencies.
mod common;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parakeet_cli::audio::resample::{StreamingResampler, resample_linear, stereo_to_mono};

fn bench_resample_one_shot(c: &mut Criterion) {
    let mut group = c.benchmark_group("resample_one_shot_48k_to_16k");

    let durations: &[(f32, &str)] = &[(1.0, "1s"), (10.0, "10s"), (60.0, "60s")];

    for &(duration, label) in durations {
        // Generate audio at 48kHz (a common source rate)
        let n_samples = (duration * 48_000.0) as usize;
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        group.throughput(Throughput::Elements(audio.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &audio, |b, samples| {
            b.iter(|| {
                let _resampled = resample_linear(samples, 48_000, 16_000);
            });
        });
    }

    group.finish();
}

fn bench_resample_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("resample_streaming_48k_to_16k");

    let durations: &[(f32, &str)] = &[(1.0, "1s"), (10.0, "10s")];

    // Typical audio callback chunk size: ~10ms at 48kHz = 480 samples
    let chunk_size = 480;

    for &(duration, label) in durations {
        let n_samples = (duration * 48_000.0) as usize;
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| {
                let t = i as f32 / 48_000.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        group.throughput(Throughput::Elements(audio.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &audio, |b, samples| {
            b.iter(|| {
                let mut resampler = StreamingResampler::new(48_000, 16_000);
                let mut output = Vec::new();
                for chunk in samples.chunks(chunk_size) {
                    output.extend(resampler.process(chunk));
                }
                output.extend(resampler.finish());
                output
            });
        });
    }

    group.finish();
}

fn bench_stereo_to_mono(c: &mut Criterion) {
    let mut group = c.benchmark_group("stereo_to_mono");

    let durations: &[(f32, &str)] = &[(1.0, "1s"), (10.0, "10s"), (60.0, "60s")];

    for &(duration, label) in durations {
        let stereo = common::generate_stereo_audio(duration, 48_000);

        group.throughput(Throughput::Elements((stereo.len() / 2) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &stereo, |b, samples| {
            b.iter(|| {
                let _mono = stereo_to_mono(samples, 2);
            });
        });
    }

    group.finish();
}

fn bench_resample_passthrough(c: &mut Criterion) {
    // Benchmark the same-rate (16k -> 16k) passthrough to verify it's ~free
    let audio = common::generate_sine_audio(10.0, 440.0);
    c.bench_function("resample_passthrough_16k_10s", |b| {
        b.iter(|| {
            let _out = resample_linear(&audio, 16_000, 16_000);
        });
    });
}

criterion_group!(
    benches,
    bench_resample_one_shot,
    bench_resample_streaming,
    bench_stereo_to_mono,
    bench_resample_passthrough,
);
criterion_main!(benches);
