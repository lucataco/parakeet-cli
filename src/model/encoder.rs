use anyhow::{Context, Result};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use std::borrow::Cow;
use std::path::Path;

pub struct Encoder {
    session: Session,
}

impl Encoder {
    pub fn load(
        path: &Path,
        use_coreml: bool,
        verbose: bool,
        cache_dir: Option<&Path>,
    ) -> Result<Self> {
        let session = if use_coreml {
            match Self::try_load_with_coreml(path, verbose, cache_dir) {
                Ok(s) => {
                    if verbose {
                        eprintln!("Encoder loaded with CoreML execution provider");
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

        if verbose {
            eprintln!("Encoder inputs/outputs:");
            for input in session.inputs() {
                eprintln!("  input: {} {:?}", input.name(), input.dtype());
            }
            for output in session.outputs() {
                eprintln!("  output: {} {:?}", output.name(), output.dtype());
            }
        }

        Ok(Self { session })
    }

    fn try_load_with_coreml(
        path: &Path,
        verbose: bool,
        cache_dir: Option<&Path>,
    ) -> std::result::Result<Session, String> {
        let mut ep = ort::ep::CoreML::default()
            .with_subgraphs(true)
            .with_compute_units(ort::ep::coreml::ComputeUnits::All);

        if let Some(dir) = cache_dir {
            std::fs::create_dir_all(dir).map_err(|e| format!("Failed to create cache dir: {e}"))?;
            let cache_path = dir.to_string_lossy().to_string();
            if verbose {
                eprintln!("CoreML model cache directory: {cache_path}");
            }
            ep = ep.with_model_cache_dir(cache_path);
        }

        let builder = Session::builder().map_err(|e| e.to_string())?;

        let builder = Self::preload_external_data(builder, path, verbose)?;

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

    fn preload_external_data(
        builder: ort::session::builder::SessionBuilder,
        model_path: &Path,
        verbose: bool,
    ) -> std::result::Result<ort::session::builder::SessionBuilder, String> {
        let data_filename = format!(
            "{}.data",
            model_path.file_name().unwrap_or_default().to_string_lossy()
        );
        let data_path = model_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(&data_filename);

        if !data_path.exists() {
            if verbose {
                eprintln!("No external data file found, model has embedded weights");
            }
            return Ok(builder);
        }

        let file_size = std::fs::metadata(&data_path)
            .map_err(|e| format!("Failed to stat external data file: {e}"))?
            .len();

        if verbose {
            eprintln!(
                "Pre-loading external data: {} ({:.2} GB)",
                data_path.display(),
                file_size as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }

        let data = std::fs::read(&data_path)
            .map_err(|e| format!("Failed to read external data file: {e}"))?;

        if verbose {
            eprintln!(
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

        let session = builder
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("Failed to load encoder model: {}", path.display()))?;
        if verbose {
            eprintln!("Encoder loaded with CPU execution provider");
        }
        Ok(session)
    }

    pub fn encode(
        &mut self,
        features: &ndarray::Array2<f32>,
    ) -> Result<(Vec<f32>, Vec<usize>, Vec<i64>)> {
        let time_steps = features.shape()[0];
        let n_mels = features.shape()[1];

        let mut input_data = vec![0.0f32; n_mels * time_steps];
        for t in 0..time_steps {
            for m in 0..n_mels {
                input_data[m * time_steps + t] = features[[t, m]];
            }
        }

        let input_tensor = Tensor::from_array(([1usize, n_mels, time_steps], input_data))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

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

        let (enc_shape, enc_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("Failed to extract encoder output tensor")?;

        let enc_shape_vec: Vec<usize> = enc_shape.iter().map(|&d| d as usize).collect();

        let (_len_shape, len_data) = outputs[1]
            .try_extract_tensor::<i64>()
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("Failed to extract encoded lengths tensor")?;

        let lengths_vec: Vec<i64> = len_data.to_vec();

        Ok((enc_data.to_vec(), enc_shape_vec, lengths_vec))
    }
}
