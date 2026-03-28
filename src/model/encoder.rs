use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

/// FastConformer encoder for Parakeet TDT.
///
/// Takes 80-bin log-mel spectrogram features and produces
/// encoder output embeddings with 8x temporal downsampling.
pub struct Encoder {
    session: Session,
}

impl Encoder {
    /// Load the encoder ONNX model with the given execution providers.
    pub fn load(path: &Path, use_coreml: bool, verbose: bool) -> Result<Self> {
        let session = if use_coreml {
            // Try CoreML first, fall back to CPU if it fails
            match Self::try_load_with_coreml(path) {
                Ok(s) => {
                    if verbose {
                        println!("Encoder loaded with CoreML execution provider");
                    }
                    s
                }
                Err(msg) => {
                    if verbose {
                        eprintln!("CoreML failed for encoder ({msg}), falling back to CPU...");
                    }
                    Self::load_cpu(path, verbose)?
                }
            }
        } else {
            Self::load_cpu(path, verbose)?
        };

        // Log model info
        if verbose {
            println!("Encoder inputs/outputs:");
            for input in session.inputs() {
                println!("  input: {} {:?}", input.name(), input.dtype());
            }
            for output in session.outputs() {
                println!("  output: {} {:?}", output.name(), output.dtype());
            }
        }

        Ok(Self { session })
    }

    fn try_load_with_coreml(path: &Path) -> std::result::Result<Session, String> {
        let builder = Session::builder().map_err(|e| e.to_string())?;
        let mut builder = builder
            .with_execution_providers([ort::ep::CoreML::default().with_subgraphs(true).build()])
            .map_err(|e| e.to_string())?;

        let session = builder.commit_from_file(path).map_err(|e| e.to_string())?;
        Ok(session)
    }

    fn load_cpu(path: &Path, verbose: bool) -> Result<Session> {
        let mut builder = Session::builder().map_err(|e| anyhow::anyhow!("{e}"))?;
        let session = builder
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("Failed to load encoder model: {}", path.display()))?;
        if verbose {
            println!("Encoder loaded with CPU execution provider");
        }
        Ok(session)
    }

    /// Run encoder inference on mel spectrogram features.
    ///
    /// # Arguments
    /// * `features` - Log-mel spectrogram of shape [time_steps, n_mels]
    ///
    /// # Returns
    /// * Encoder output as flat vec with shape info [1, time_steps/8, hidden_dim]
    /// * Encoded lengths
    pub fn encode(
        &mut self,
        features: &ndarray::Array2<f32>,
    ) -> Result<(Vec<f32>, Vec<usize>, Vec<i64>)> {
        let time_steps = features.shape()[0];
        let n_mels = features.shape()[1];

        // Model expects [batch, n_mels, time] — we need to transpose and add batch dim
        let mut input_data = vec![0.0f32; n_mels * time_steps];
        for t in 0..time_steps {
            for m in 0..n_mels {
                input_data[m * time_steps + t] = features[[t, m]];
            }
        }

        let input_tensor = Tensor::from_array(([1usize, n_mels, time_steps], input_data))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Length tensor [batch]
        let length_tensor = Tensor::from_array(([1usize], vec![time_steps as i64]))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "audio_signal" => input_tensor,
                "length" => length_tensor,
            ])
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("Encoder inference failed")?;

        // Extract encoder output: [batch, hidden_dim=1024, time/8]
        let (enc_shape, enc_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("Failed to extract encoder output tensor")?;

        let enc_shape_vec: Vec<usize> = enc_shape.iter().map(|&d| d as usize).collect();

        // Extract encoded lengths
        let (_len_shape, len_data) = outputs[1]
            .try_extract_tensor::<i64>()
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("Failed to extract encoded lengths tensor")?;

        let lengths_vec: Vec<i64> = len_data.to_vec();

        Ok((enc_data.to_vec(), enc_shape_vec, lengths_vec))
    }
}
