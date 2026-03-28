pub mod decoder;
pub mod encoder;
pub mod tokenizer;

use anyhow::{Context, Result};
use ndarray::Array2;
use std::path::Path;

use decoder::TdtDecoder;
use encoder::Encoder;
use tokenizer::Tokenizer;

/// Configuration loaded from config.json
#[derive(serde::Deserialize, Debug)]
#[allow(dead_code)]
pub struct ModelConfig {
    pub model_type: String,
    pub features_size: usize,
    pub subsampling_factor: usize,
}

/// Complete Parakeet TDT model (encoder + decoder + tokenizer).
pub struct ParakeetModel {
    pub encoder: Encoder,
    pub decoder: TdtDecoder,
    pub tokenizer: Tokenizer,
}

impl ParakeetModel {
    /// Load the complete model from a directory.
    ///
    /// Expects the directory to contain:
    /// - encoder-model.onnx (+ .data file)
    /// - decoder_joint-model.onnx
    /// - vocab.txt
    /// - config.json
    pub fn load(model_dir: &Path, use_coreml: bool, verbose: bool) -> Result<Self> {
        if verbose {
            println!("Loading Parakeet TDT model from: {}", model_dir.display());
            println!();
        }

        // Load config
        let config_path = model_dir.join("config.json");
        let config: ModelConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config: {}", config_path.display()))?,
        )
        .context("Failed to parse config.json")?;
        if verbose {
            println!("Model config: {config:?}");
        }

        // Detect whether INT8 or FP32 files are present
        let has_fp32 = model_dir.join("encoder-model.onnx").exists();
        let has_int8 = model_dir.join("encoder-model.int8.onnx").exists();

        let (encoder_path, decoder_path) = if has_fp32 {
            if verbose {
                println!("Using FP32 model");
            }
            (
                model_dir.join("encoder-model.onnx"),
                model_dir.join("decoder_joint-model.onnx"),
            )
        } else if has_int8 {
            if verbose {
                println!("Using INT8 quantized model");
            }
            (
                model_dir.join("encoder-model.int8.onnx"),
                model_dir.join("decoder_joint-model.int8.onnx"),
            )
        } else {
            anyhow::bail!(
                "No model files found in {}. Run `parakeet download` first.",
                model_dir.display()
            );
        };

        // Load tokenizer
        let vocab_path = model_dir.join("vocab.txt");
        let tokenizer = Tokenizer::from_file(&vocab_path, verbose)?;

        // Load encoder (with CoreML if available)
        if verbose {
            println!();
        }
        let encoder = Encoder::load(&encoder_path, use_coreml, verbose)?;

        // Load decoder (CPU only — decoder is small and autoregressive)
        if verbose {
            println!();
        }
        let decoder = TdtDecoder::load(&decoder_path, verbose)?;

        if verbose {
            println!();
            println!("Model loaded successfully!");
        }

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
        })
    }

    /// Transcribe audio from mel spectrogram features.
    ///
    /// # Arguments
    /// * `features` - Log-mel spectrogram of shape [time_steps, n_mels]
    ///
    /// # Returns
    /// * Transcribed text
    pub fn transcribe(&mut self, features: &Array2<f32>) -> Result<String> {
        // Run encoder
        let (enc_output, enc_shape, lengths) = self.encoder.encode(features)?;
        let encoded_length = lengths[0];

        // Run TDT greedy decoding
        let token_ids = self.decoder.decode_greedy(
            &enc_output,
            &enc_shape,
            encoded_length,
            self.tokenizer.blank_id,
        )?;

        // Decode tokens to text
        let text = self.tokenizer.decode(&token_ids);

        Ok(text)
    }
}
