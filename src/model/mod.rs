pub mod decoder;
pub mod encoder;
pub mod tokenizer;

use anyhow::{Context, Result};
use ndarray::Array2;
use std::path::{Path, PathBuf};

use decoder::TdtDecoder;
use encoder::Encoder;
use tokenizer::Tokenizer;

/// Configuration loaded from config.json
#[derive(serde::Deserialize, Debug)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelVariant {
    Fp16,
    Int8,
    Fp32,
}

impl ModelVariant {
    fn name(self) -> &'static str {
        match self {
            Self::Fp16 => "FP16",
            Self::Int8 => "INT8",
            Self::Fp32 => "FP32 (legacy)",
        }
    }

    fn encoder_path(self, model_dir: &Path) -> PathBuf {
        model_dir.join(match self {
            Self::Fp16 => "encoder-model.fp16.onnx",
            Self::Int8 => "encoder-model.int8.onnx",
            Self::Fp32 => "encoder-model.onnx",
        })
    }

    fn decoder_path(self, model_dir: &Path) -> PathBuf {
        model_dir.join(match self {
            Self::Fp16 => "decoder_joint-model.fp16.onnx",
            Self::Int8 => "decoder_joint-model.int8.onnx",
            Self::Fp32 => "decoder_joint-model.onnx",
        })
    }

    fn is_available(self, model_dir: &Path) -> bool {
        self.encoder_path(model_dir).exists() && self.decoder_path(model_dir).exists()
    }
}

fn preferred_variant(model_dir: &Path) -> Option<ModelVariant> {
    let marker = std::fs::read_to_string(model_dir.join(".variant")).ok()?;
    let marker = marker.to_ascii_lowercase();

    if marker.contains("int8") {
        Some(ModelVariant::Int8)
    } else if marker.contains("fp16") {
        Some(ModelVariant::Fp16)
    } else if marker.contains("fp32") {
        Some(ModelVariant::Fp32)
    } else {
        None
    }
}

fn select_model_variant(model_dir: &Path) -> Option<ModelVariant> {
    if let Some(variant) = preferred_variant(model_dir) {
        if variant.is_available(model_dir) {
            return Some(variant);
        }
    }

    [ModelVariant::Fp16, ModelVariant::Int8, ModelVariant::Fp32]
        .into_iter()
        .find(|variant| variant.is_available(model_dir))
}

impl ParakeetModel {
    /// Load the complete model from a directory.
    ///
    /// Detects model variant automatically in priority order:
    /// FP16 > INT8 > FP32 (legacy).
    ///
    /// Expects the directory to contain:
    /// - encoder model (one of: .fp16.onnx, .int8.onnx, .onnx)
    /// - decoder model (matching variant)
    /// - vocab.txt
    /// - config.json
    pub fn load(model_dir: &Path, use_coreml: bool, verbose: bool) -> Result<Self> {
        if verbose {
            eprintln!("Loading Parakeet TDT model from: {}", model_dir.display());
            eprintln!();
        }

        // Load config
        let config_path = model_dir.join("config.json");
        let config: ModelConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config: {}", config_path.display()))?,
        )
        .context("Failed to parse config.json")?;
        if verbose {
            eprintln!("Model config: {config:?}");
        }

        let variant = select_model_variant(model_dir).with_context(|| {
            format!(
                "No complete model files found in {}. Run `parakeet download` first.",
                model_dir.display()
            )
        })?;
        let encoder_path = variant.encoder_path(model_dir);
        let decoder_path = variant.decoder_path(model_dir);

        if verbose {
            eprintln!("Using {} model", variant.name());
        }

        // Load tokenizer first -- we need vocab_size for the decoder
        let vocab_path = model_dir.join("vocab.txt");
        let tokenizer = Tokenizer::from_file(&vocab_path, verbose)?;
        let vocab_size = tokenizer.vocab_size();

        // Load encoder (with CoreML if requested)
        let coreml_cache = if use_coreml {
            Some(model_dir.join("coreml_cache"))
        } else {
            None
        };

        if verbose {
            eprintln!();
        }
        let encoder = Encoder::load(&encoder_path, use_coreml, verbose, coreml_cache.as_deref())?;

        // Load decoder (CPU only -- decoder is small and autoregressive)
        if verbose {
            eprintln!();
        }
        let decoder = TdtDecoder::load(&decoder_path, vocab_size, verbose)?;

        if verbose {
            eprintln!();
            eprintln!("Model loaded successfully!");
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
        let encoded_length = *lengths
            .first()
            .context("Encoder returned no output lengths")?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_model_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "parakeet-model-selection-{}-{unique}",
            std::process::id()
        ))
    }

    fn touch(path: &Path) {
        fs::write(path, []).unwrap();
    }

    fn touch_variant_pair(dir: &Path, variant: ModelVariant) {
        touch(&variant.encoder_path(dir));
        touch(&variant.decoder_path(dir));
    }

    #[test]
    fn model_selection_ignores_incomplete_higher_priority_variant() {
        let dir = temp_model_dir();
        fs::create_dir_all(&dir).unwrap();
        touch(&ModelVariant::Fp16.encoder_path(&dir));
        touch_variant_pair(&dir, ModelVariant::Int8);

        assert_eq!(select_model_variant(&dir), Some(ModelVariant::Int8));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn model_selection_honors_variant_marker_when_available() {
        let dir = temp_model_dir();
        fs::create_dir_all(&dir).unwrap();
        touch_variant_pair(&dir, ModelVariant::Fp16);
        touch_variant_pair(&dir, ModelVariant::Int8);
        fs::write(dir.join(".variant"), "INT8 quantized").unwrap();

        assert_eq!(select_model_variant(&dir), Some(ModelVariant::Int8));

        fs::remove_dir_all(dir).unwrap();
    }
}
