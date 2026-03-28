use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::borrow::Cow;
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
    ///
    /// When `use_coreml` is true, the encoder is loaded with an aggressively
    /// tuned CoreML execution provider targeting the Apple Neural Engine,
    /// with compiled-model caching for fast subsequent loads.
    pub fn load(
        path: &Path,
        use_coreml: bool,
        verbose: bool,
        cache_dir: Option<&Path>,
    ) -> Result<Self> {
        let session = if use_coreml {
            // Try CoreML first, fall back to CPU if it fails
            match Self::try_load_with_coreml(path, verbose, cache_dir) {
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

    fn try_load_with_coreml(
        path: &Path,
        verbose: bool,
        cache_dir: Option<&Path>,
    ) -> std::result::Result<Session, String> {
        // CoreML execution provider configuration.
        //
        // ComputeUnits::All lets CoreML dispatch to ANE + GPU + CPU.
        // FP16 models are natively supported by Apple Silicon's ANE,
        // which should provide significant speedups over CPU-only inference.
        //
        // NeuralNetwork format (default, CoreML 3+) is used instead of
        // MLProgram because the encoder has dynamic time dimensions that
        // cause MLProgram compilation to fail with error code -14.
        //
        // ModelCacheDirectory caches the compiled CoreML model on disk
        // so subsequent session loads skip the ONNX->CoreML compilation.
        let mut ep = ort::ep::CoreML::default()
            .with_subgraphs(true)
            .with_compute_units(ort::ep::coreml::ComputeUnits::All);

        if let Some(dir) = cache_dir {
            std::fs::create_dir_all(dir).map_err(|e| format!("Failed to create cache dir: {e}"))?;
            let cache_path = dir.to_string_lossy().to_string();
            if verbose {
                println!("CoreML model cache directory: {cache_path}");
            }
            ep = ep.with_model_cache_dir(cache_path);
        }

        let builder = Session::builder().map_err(|e| e.to_string())?;

        // If the model uses external data (e.g. encoder-model.onnx.data),
        // pre-load it into memory so ONNX Runtime can resolve the external
        // tensor references without filesystem path issues. The CoreML EP
        // has a known bug where it misresolves external data file paths,
        // treating the .onnx file as a directory. Pre-loading via
        // with_external_initializer_file_in_memory() bypasses this entirely.
        let builder = Self::preload_external_data(builder, path, verbose)?;

        // Session-level optimizations
        let builder = builder
            .with_execution_providers([ep.build()])
            .map_err(|e| e.to_string())?;
        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| e.to_string())?;
        let mut builder = builder
            .with_memory_pattern(true)
            .map_err(|e| e.to_string())?;

        let session = builder.commit_from_file(path).map_err(|e| e.to_string())?;
        Ok(session)
    }

    /// Pre-load external data files into memory for the session builder.
    ///
    /// ONNX models with external data store their weights in a separate
    /// file (e.g. `encoder-model.onnx.data`). The ONNX Runtime CoreML EP
    /// has path resolution issues with these files. By reading the data
    /// into memory and registering it with `with_external_initializer_file_in_memory`,
    /// we let ONNX Runtime access the weights without touching the filesystem.
    fn preload_external_data(
        builder: ort::session::builder::SessionBuilder,
        model_path: &Path,
        verbose: bool,
    ) -> std::result::Result<ort::session::builder::SessionBuilder, String> {
        // The external data file is typically named <model>.data
        // e.g. encoder-model.onnx -> encoder-model.onnx.data
        let data_filename = format!(
            "{}.data",
            model_path.file_name().unwrap_or_default().to_string_lossy()
        );
        let data_path = model_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(&data_filename);

        if !data_path.exists() {
            // No external data file -- model has embedded weights, nothing to do
            if verbose {
                println!("No external data file found, model has embedded weights");
            }
            return Ok(builder);
        }

        let file_size = std::fs::metadata(&data_path)
            .map_err(|e| format!("Failed to stat external data file: {e}"))?
            .len();

        if verbose {
            println!(
                "Pre-loading external data: {} ({:.2} GB)",
                data_path.display(),
                file_size as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }

        let data = std::fs::read(&data_path)
            .map_err(|e| format!("Failed to read external data file: {e}"))?;

        if verbose {
            println!(
                "External data loaded into memory ({:.2} GB)",
                data.len() as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }

        let builder = builder
            .with_external_initializer_file_in_memory(&data_filename, Cow::Owned(data))
            .map_err(|e| e.to_string())?;

        Ok(builder)
    }

    fn load_cpu(path: &Path, verbose: bool) -> Result<Session> {
        let builder = Session::builder().map_err(|e| anyhow::anyhow!("{e}"))?;
        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut builder = builder
            .with_memory_pattern(true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // CPU-only path: let ONNX Runtime use its default thread count
        // (all logical cores). The encoder's large matrix multiplications
        // benefit from maximum parallelism, even on efficiency cores.

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
