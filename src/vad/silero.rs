use crate::integrity::file_matches_sha256;
use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::{Tensor, ValueType};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

pub const SILERO_VAD_URL: &str = "https://raw.githubusercontent.com/snakers4/silero-vad/980b17e9d56463e51393a8d92ded473f1b17896a/src/silero_vad/data/silero_vad.onnx";
const SILERO_VAD_SHA256: &str = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3";
const USER_AGENT: &str = concat!("parakeet-cli/", env!("CARGO_PKG_VERSION"));

pub const VAD_CHUNK_SAMPLES: usize = 512;
pub const VAD_CONTEXT_SAMPLES: usize = 64;

pub const VAD_SAMPLE_RATE: u32 = 16000;

pub struct SileroVad {
    session: Session,
    state: Vec<f32>,
    state_dim: usize,
    sr: i64,
    context: Vec<f32>,
    logged_io_shapes: bool,
    verbose: bool,
}

impl SileroVad {
    pub fn load(path: &std::path::Path, verbose: bool) -> Result<Self> {
        let mut builder = Session::builder().map_err(|e| anyhow::anyhow!("{e}"))?;
        let session = builder
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("Failed to load Silero VAD: {}", path.display()))?;

        let input_dtype = validate_tensor_outlet(session.inputs(), "input", 2)?;
        let state_dtype = validate_tensor_outlet(session.inputs(), "state", 3)?;
        let sr_dtype = validate_tensor_outlet(session.inputs(), "sr", 0)?;
        let output_dtype = validate_tensor_outlet(session.outputs(), "output", 2)?;
        let _state_out_dtype = validate_tensor_outlet(session.outputs(), "stateN", 3)?;

        if verbose {
            eprintln!("Silero VAD loaded:");
            for input in session.inputs() {
                eprintln!("  input: {} {:?}", input.name(), input.dtype());
            }
            for output in session.outputs() {
                eprintln!("  output: {} {:?}", output.name(), output.dtype());
            }
        }

        let state_dim = extract_concrete_state_dim(state_dtype)?;

        if verbose {
            eprintln!(
                "Silero VAD contract: input={} state={} sr={} output={}",
                describe_tensor_shape(input_dtype),
                describe_tensor_shape(state_dtype),
                describe_tensor_shape(sr_dtype),
                describe_tensor_shape(output_dtype),
            );
        }

        let state = vec![0.0f32; 2 * state_dim];

        Ok(Self {
            session,
            state,
            state_dim,
            sr: VAD_SAMPLE_RATE as i64,
            context: vec![0.0f32; VAD_CONTEXT_SAMPLES],
            logged_io_shapes: false,
            verbose,
        })
    }

    pub fn reset(&mut self) {
        reset_state_and_context(&mut self.state, self.state_dim, &mut self.context);
    }

    pub fn process_chunk(&mut self, chunk: &[f32]) -> Result<f32> {
        if chunk.len() != VAD_CHUNK_SAMPLES {
            anyhow::bail!(
                "VAD chunk must be exactly {} samples, got {}",
                VAD_CHUNK_SAMPLES,
                chunk.len()
            );
        }

        let model_input = build_model_input(&self.context, chunk);
        let model_input_len = model_input.len();

        let input_tensor = Tensor::from_array(([1usize, model_input_len], model_input.clone()))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let state_tensor =
            Tensor::from_array(([2usize, 1usize, self.state_dim], self.state.clone()))
                .map_err(|e| anyhow::anyhow!("{e}"))?;

        let sr_tensor =
            Tensor::from_array(((), vec![self.sr])).map_err(|e| anyhow::anyhow!("{e}"))?;

        if self.verbose && !self.logged_io_shapes {
            eprintln!(
                "[debug] Silero inputs: input=[1, {}] state=[2, 1, {}] sr=[]",
                model_input_len, self.state_dim
            );
        }

        let outputs = self
            .session
            .run(ort::inputs![
                "input" => input_tensor,
                "state" => state_tensor,
                "sr" => sr_tensor,
            ])
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("Silero VAD inference failed")?;

        let (_prob_shape, prob_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let speech_prob = *prob_data
            .first()
            .context("Silero VAD returned an empty probability tensor")?;

        let (_state_shape, state_data) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if self.verbose && !self.logged_io_shapes {
            eprintln!("[debug] Silero output stateN shape: {:?}", _state_shape);
            self.logged_io_shapes = true;
        }
        if state_data.len() != self.state.len() {
            anyhow::bail!(
                "Silero VAD returned invalid state length: got {}, expected {}",
                state_data.len(),
                self.state.len(),
            );
        }
        self.state = state_data.to_vec();
        self.context
            .copy_from_slice(&model_input[model_input_len - VAD_CONTEXT_SAMPLES..]);

        Ok(speech_prob)
    }
}

fn build_model_input(context: &[f32], chunk: &[f32]) -> Vec<f32> {
    let mut input = Vec::with_capacity(context.len() + chunk.len());
    input.extend_from_slice(context);
    input.extend_from_slice(chunk);
    input
}

fn reset_state_and_context(state: &mut Vec<f32>, state_dim: usize, context: &mut [f32]) {
    *state = vec![0.0f32; 2 * state_dim];
    context.fill(0.0);
}

fn validate_tensor_outlet<'a>(
    outlets: &'a [ort::value::Outlet],
    name: &str,
    expected_rank: usize,
) -> Result<&'a ValueType> {
    let outlet = outlets
        .iter()
        .find(|o| o.name() == name)
        .with_context(|| format!("Silero VAD is missing required tensor '{name}'"))?;

    let dtype = outlet.dtype();
    match dtype {
        ValueType::Tensor { shape, .. } if shape.len() == expected_rank => Ok(dtype),
        ValueType::Tensor { shape, .. } => anyhow::bail!(
            "Silero tensor '{name}' has rank {}, expected {} (shape {})",
            shape.len(),
            expected_rank,
            describe_tensor_shape(dtype),
        ),
        _ => anyhow::bail!("Silero outlet '{name}' is not a tensor: {dtype:?}"),
    }
}

fn extract_concrete_state_dim(dtype: &ValueType) -> Result<usize> {
    let shape = dtype
        .tensor_shape()
        .context("Silero state input is not a tensor")?;

    if shape.len() != 3 {
        anyhow::bail!(
            "Silero state input must have rank 3, got shape {}",
            describe_tensor_shape(dtype)
        );
    }

    let hidden_dim = shape[2];
    if hidden_dim <= 0 {
        anyhow::bail!(
            "Silero state hidden dimension must be concrete, got shape {}",
            describe_tensor_shape(dtype)
        );
    }

    Ok(hidden_dim as usize)
}

fn describe_tensor_shape(dtype: &ValueType) -> String {
    match dtype.tensor_shape() {
        Some(shape) => format!("{shape}"),
        None => "<non-tensor>".to_string(),
    }
}

pub async fn ensure_vad_model(model_dir: &std::path::Path) -> Result<std::path::PathBuf> {
    let vad_path = model_dir.join("silero_vad.onnx");

    if vad_path.exists() && file_matches_sha256(&vad_path, SILERO_VAD_SHA256)? {
        return Ok(vad_path);
    }

    if vad_path.exists() {
        eprintln!("Silero VAD checksum mismatch, re-downloading...");
        tokio::fs::remove_file(&vad_path).await.with_context(|| {
            format!(
                "Failed to remove invalid cached Silero VAD model: {}",
                vad_path.display()
            )
        })?;
    } else {
        eprintln!("Downloading Silero VAD model...");
    }

    tokio::fs::create_dir_all(model_dir).await?;

    let client = reqwest::Client::builder()
        .https_only(true)
        .user_agent(USER_AGENT)
        .build()?;

    let response = client
        .get(SILERO_VAD_URL)
        .send()
        .await
        .context("Failed to download Silero VAD")?;

    if !response.status().is_success() {
        anyhow::bail!("Failed to download Silero VAD: HTTP {}", response.status());
    }

    let bytes = response.bytes().await?;
    let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if actual_sha256 != SILERO_VAD_SHA256 {
        anyhow::bail!(
            "Silero VAD checksum mismatch: expected {}, got {}",
            SILERO_VAD_SHA256,
            actual_sha256,
        );
    }

    let tmp_path = model_dir.join(".silero_vad.onnx.tmp");
    let _ = tokio::fs::remove_file(&tmp_path).await;
    let mut file = tokio::fs::File::create(&tmp_path).await?;
    file.write_all(&bytes).await?;
    file.sync_all().await?;
    drop(file);
    tokio::fs::rename(&tmp_path, &vad_path).await?;

    eprintln!(
        "Silero VAD downloaded ({} bytes) to {}",
        bytes.len(),
        vad_path.display()
    );

    Ok(vad_path)
}

pub struct VadSegmenter {
    threshold: f32,
    silence_chunks_needed: usize,
    state: VadState,
    silence_count: usize,
    min_speech_chunks: usize,
    speech_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VadState {
    Silence,
    Speaking,
}

#[derive(Debug)]
pub enum VadEvent {
    None,
    SpeechStart,
    SpeechEnd,
}

impl VadSegmenter {
    pub fn new(threshold: f32, silence_ms: u64) -> Self {
        let chunk_ms = (VAD_CHUNK_SAMPLES as f64 / VAD_SAMPLE_RATE as f64 * 1000.0) as u64;
        let silence_chunks_needed = silence_ms.div_ceil(chunk_ms).max(1) as usize;

        let min_speech_chunks = (100 / chunk_ms).max(1) as usize;

        Self {
            threshold,
            silence_chunks_needed,
            state: VadState::Silence,
            silence_count: 0,
            min_speech_chunks,
            speech_count: 0,
        }
    }

    pub fn process(&mut self, speech_prob: f32) -> VadEvent {
        let is_speech = speech_prob >= self.threshold;

        match self.state {
            VadState::Silence => {
                if is_speech {
                    self.state = VadState::Speaking;
                    self.silence_count = 0;
                    self.speech_count = 1;
                    VadEvent::SpeechStart
                } else {
                    VadEvent::None
                }
            }
            VadState::Speaking => {
                if is_speech {
                    self.silence_count = 0;
                    self.speech_count += 1;
                    VadEvent::None
                } else {
                    self.silence_count += 1;
                    if self.silence_count >= self.silence_chunks_needed {
                        self.state = VadState::Silence;
                        let was_valid = self.speech_count >= self.min_speech_chunks;
                        self.speech_count = 0;
                        self.silence_count = 0;
                        if was_valid {
                            VadEvent::SpeechEnd
                        } else {
                            VadEvent::None
                        }
                    } else {
                        VadEvent::None
                    }
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.state = VadState::Silence;
        self.silence_count = 0;
        self.speech_count = 0;
    }

    pub fn state(&self) -> VadState {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ort::value::{Shape, SymbolicDimensions, TensorElementType};

    fn tensor_dtype(shape: &[i64]) -> ValueType {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape: Shape::from(shape),
            dimension_symbols: SymbolicDimensions::empty(shape.len()),
        }
    }

    #[test]
    fn test_vad_segmenter_basic() {
        let mut seg = VadSegmenter::new(0.5, 500);

        assert!(matches!(seg.process(0.1), VadEvent::None));
        assert_eq!(seg.state(), VadState::Silence);

        assert!(matches!(seg.process(0.8), VadEvent::SpeechStart));
        assert_eq!(seg.state(), VadState::Speaking);

        for _ in 0..10 {
            assert!(matches!(seg.process(0.9), VadEvent::None));
        }

        for _ in 0..5 {
            assert!(matches!(seg.process(0.1), VadEvent::None));
            assert_eq!(seg.state(), VadState::Speaking);
        }

        let mut ended = false;
        for _ in 0..20 {
            if matches!(seg.process(0.1), VadEvent::SpeechEnd) {
                ended = true;
                break;
            }
        }
        assert!(ended);
        assert_eq!(seg.state(), VadState::Silence);
    }

    #[test]
    fn test_vad_segmenter_ceil_silence_timeout() {
        let mut seg = VadSegmenter::new(0.5, 33);

        assert!(matches!(seg.process(0.8), VadEvent::SpeechStart));
        assert!(matches!(seg.process(0.8), VadEvent::None));
        assert!(matches!(seg.process(0.8), VadEvent::None));

        assert!(matches!(seg.process(0.1), VadEvent::None));
        assert_eq!(seg.state(), VadState::Speaking);
        assert!(matches!(seg.process(0.1), VadEvent::SpeechEnd));
        assert_eq!(seg.state(), VadState::Silence);
    }

    #[test]
    fn test_extract_concrete_state_dim() {
        let dtype = tensor_dtype(&[2, -1, 128]);
        assert_eq!(extract_concrete_state_dim(&dtype).unwrap(), 128);
    }

    #[test]
    fn test_extract_concrete_state_dim_rejects_dynamic_hidden() {
        let dtype = tensor_dtype(&[2, -1, -1]);
        assert!(extract_concrete_state_dim(&dtype).is_err());
    }

    #[test]
    fn test_describe_tensor_shape_scalar() {
        let dtype = tensor_dtype(&[]);
        assert_eq!(describe_tensor_shape(&dtype), "[]");
    }

    #[test]
    fn test_build_model_input_prepends_context() {
        let context = vec![1.0; VAD_CONTEXT_SAMPLES];
        let chunk = vec![2.0; VAD_CHUNK_SAMPLES];
        let input = build_model_input(&context, &chunk);

        assert_eq!(input.len(), VAD_CONTEXT_SAMPLES + VAD_CHUNK_SAMPLES);
        assert!(input[..VAD_CONTEXT_SAMPLES].iter().all(|&x| x == 1.0));
        assert!(input[VAD_CONTEXT_SAMPLES..].iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_reset_state_and_context() {
        let mut state = vec![1.0; 2 * 128];
        let mut context = vec![1.0; VAD_CONTEXT_SAMPLES];

        reset_state_and_context(&mut state, 128, &mut context);

        assert_eq!(state.len(), 2 * 128);
        assert!(state.iter().all(|&x| x == 0.0));
        assert!(context.iter().all(|&x| x == 0.0));
    }
}
