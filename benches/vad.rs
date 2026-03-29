/// Benchmarks for Silero Voice Activity Detection.
///
/// Measures per-chunk inference latency and streaming throughput.
/// The VAD processes 512-sample chunks (32ms at 16kHz), so it must
/// be faster than 32ms per chunk to keep up with realtime audio.
///
/// Also benchmarks the pure VadSegmenter state machine (no ONNX).
mod common;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parakeet_cli::vad::silero::{SileroVad, VAD_CHUNK_SAMPLES, VadSegmenter};

fn bench_vad_single_chunk(c: &mut Criterion) {
    if !common::vad_model_available() {
        eprintln!(
            "SKIP: VAD benchmarks require silero_vad.onnx in model dir. \
             Run `parakeet listen` once to auto-download it."
        );
        return;
    }

    let model_dir = common::default_model_dir();
    let vad_path = model_dir.join("silero_vad.onnx");

    let mut vad = SileroVad::load(&vad_path, false).expect("Failed to load Silero VAD");

    // Single 32ms chunk
    let chunk = vec![0.0f32; VAD_CHUNK_SAMPLES];

    c.bench_function("vad_single_chunk_32ms", |b| {
        b.iter(|| {
            let _prob = vad.process_chunk(&chunk).expect("VAD inference failed");
        });
    });
}

fn bench_vad_stream(c: &mut Criterion) {
    if !common::vad_model_available() {
        return;
    }

    let model_dir = common::default_model_dir();
    let vad_path = model_dir.join("silero_vad.onnx");

    let durations: &[(f32, &str)] = &[(1.0, "1s"), (5.0, "5s"), (10.0, "10s")];

    let mut group = c.benchmark_group("vad_stream");
    group.sample_size(10);

    for &(duration, label) in durations {
        let audio = common::generate_sine_audio(duration, 440.0);
        let n_chunks = audio.len() / VAD_CHUNK_SAMPLES;

        group.throughput(Throughput::Elements(n_chunks as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &audio, |b, samples| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let mut vad = SileroVad::load(&vad_path, false).expect("Failed to load VAD");
                    let start = std::time::Instant::now();
                    for chunk in samples.chunks_exact(VAD_CHUNK_SAMPLES) {
                        let _prob = vad.process_chunk(chunk).expect("VAD failed");
                    }
                    total += start.elapsed();
                }
                total
            });
        });
    }

    group.finish();
}

fn bench_vad_segmenter_throughput(c: &mut Criterion) {
    // Pure state machine benchmark — no ONNX, no model dependency.
    // Simulates a stream of speech probabilities through the segmenter.
    let n_chunks = 10_000; // ~320 seconds worth of VAD decisions

    // Create a realistic probability sequence:
    // silence -> speech -> silence -> speech -> ...
    let mut probs = Vec::with_capacity(n_chunks);
    for i in 0..n_chunks {
        let cycle = i % 200; // ~6.4s cycle
        if cycle < 100 {
            // Speech phase: high probability
            probs.push(0.85);
        } else {
            // Silence phase: low probability
            probs.push(0.05);
        }
    }

    let mut group = c.benchmark_group("vad_segmenter");
    group.throughput(Throughput::Elements(n_chunks as u64));

    group.bench_function("state_machine_10k_chunks", |b| {
        b.iter(|| {
            let mut seg = VadSegmenter::new(0.5, 1500);
            let mut events = 0u32;
            for &prob in &probs {
                let event = seg.process(prob);
                if matches!(event, parakeet_cli::vad::VadEvent::SpeechEnd) {
                    events += 1;
                }
            }
            events
        });
    });

    group.finish();
}

fn bench_vad_latency_vs_realtime(c: &mut Criterion) {
    // Benchmark that explicitly compares VAD chunk processing time
    // against the 32ms realtime budget. This helps determine if
    // VAD can keep up with live audio.
    if !common::vad_model_available() {
        return;
    }

    let model_dir = common::default_model_dir();
    let vad_path = model_dir.join("silero_vad.onnx");

    let mut vad = SileroVad::load(&vad_path, false).expect("Failed to load Silero VAD");

    // Mix of silence and speech-like content
    let speech_chunk: Vec<f32> = (0..VAD_CHUNK_SAMPLES)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            (2.0 * std::f32::consts::PI * 300.0 * t).sin() * 0.3
                + (2.0 * std::f32::consts::PI * 1200.0 * t).sin() * 0.1
        })
        .collect();

    let silence_chunk = vec![0.0f32; VAD_CHUNK_SAMPLES];

    let mut group = c.benchmark_group("vad_latency");

    group.bench_function("speech_chunk", |b| {
        b.iter(|| {
            let _prob = vad.process_chunk(&speech_chunk).expect("VAD failed");
        });
    });

    group.bench_function("silence_chunk", |b| {
        b.iter(|| {
            let _prob = vad.process_chunk(&silence_chunk).expect("VAD failed");
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_vad_single_chunk,
    bench_vad_stream,
    bench_vad_segmenter_throughput,
    bench_vad_latency_vs_realtime,
);
criterion_main!(benches);
