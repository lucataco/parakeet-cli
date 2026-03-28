/// Benchmarks for the TDT greedy decoder.
///
/// The decoder runs one ONNX session per time-step in an autoregressive
/// loop. We measure total decode time and per-step latency for various
/// encoder output lengths.
///
/// Requires model files to be downloaded (`parakeet download`).
mod common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use parakeet_cli::audio::{compute_mel_spectrogram, MelConfig};
use parakeet_cli::model::decoder::TdtDecoder;
use parakeet_cli::model::encoder::Encoder;
use parakeet_cli::model::tokenizer::Tokenizer;

/// Pre-computed encoder output for a given duration, ready for decoder benchmarks.
struct EncoderOutput {
    data: Vec<f32>,
    shape: Vec<usize>,
    encoded_length: i64,
}

fn prepare_encoder_output(encoder: &mut Encoder, duration_secs: f32) -> EncoderOutput {
    let config = MelConfig::default();
    let audio = common::generate_sine_audio(duration_secs, 440.0);
    let features = compute_mel_spectrogram(&audio, &config);
    let (data, shape, lengths) = encoder.encode(&features).expect("Encoder failed");
    EncoderOutput {
        data,
        shape,
        encoded_length: lengths[0],
    }
}

fn bench_decoder_greedy(c: &mut Criterion) {
    if !common::model_available() {
        eprintln!("SKIP: decoder benchmarks require model files. Run `parakeet download` first.");
        return;
    }

    let model_dir = common::default_model_dir();
    let encoder_path = common::encoder_path();
    let decoder_path = common::decoder_path();
    let vocab_path = model_dir.join("vocab.txt");
    let cache_dir = model_dir.join("coreml_cache");

    let mut encoder = Encoder::load(&encoder_path, false, false, Some(&cache_dir))
        .expect("Failed to load encoder");
    let tokenizer = Tokenizer::from_file(&vocab_path, false).expect("Failed to load tokenizer");
    let mut decoder = TdtDecoder::load(&decoder_path, tokenizer.vocab_size(), false)
        .expect("Failed to load decoder");

    let durations: &[(f32, &str)] = &[(1.0, "1s"), (5.0, "5s"), (10.0, "10s"), (30.0, "30s")];

    let mut group = c.benchmark_group("decoder_greedy");
    group.sample_size(10);

    for &(duration, label) in durations {
        // Pre-compute encoder output outside the timed region
        let enc_out = prepare_encoder_output(&mut encoder, duration);

        group.bench_with_input(BenchmarkId::from_parameter(label), &enc_out, |b, enc| {
            b.iter(|| {
                let tokens = decoder
                    .decode_greedy(
                        &enc.data,
                        &enc.shape,
                        enc.encoded_length,
                        tokenizer.blank_id,
                    )
                    .expect("Decoder failed");
                tokens
            });
        });
    }

    group.finish();
}

fn bench_decoder_with_tokenizer(c: &mut Criterion) {
    if !common::model_available() {
        return;
    }

    let model_dir = common::default_model_dir();
    let encoder_path = common::encoder_path();
    let decoder_path = common::decoder_path();
    let vocab_path = model_dir.join("vocab.txt");
    let cache_dir = model_dir.join("coreml_cache");

    let mut encoder = Encoder::load(&encoder_path, false, false, Some(&cache_dir))
        .expect("Failed to load encoder");
    let tokenizer = Tokenizer::from_file(&vocab_path, false).expect("Failed to load tokenizer");
    let mut decoder = TdtDecoder::load(&decoder_path, tokenizer.vocab_size(), false)
        .expect("Failed to load decoder");

    // Measure decoder + tokenizer combined for 10s audio
    let enc_out = prepare_encoder_output(&mut encoder, 10.0);

    let mut group = c.benchmark_group("decoder_plus_tokenizer");
    group.sample_size(10);

    group.bench_function("10s", |b| {
        b.iter(|| {
            let tokens = decoder
                .decode_greedy(
                    &enc_out.data,
                    &enc_out.shape,
                    enc_out.encoded_length,
                    tokenizer.blank_id,
                )
                .expect("Decoder failed");
            let _text = tokenizer.decode(&tokens);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_decoder_greedy, bench_decoder_with_tokenizer);
criterion_main!(benches);
