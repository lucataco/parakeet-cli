/// Silero Voice Activity Detection (VAD) via ONNX Runtime.
///
/// Uses the Silero VAD v5 model to detect speech segments in audio.
/// The model processes 512-sample chunks (32ms at 16kHz) and outputs
/// a speech probability for each chunk.
///
/// The VAD maintains internal state (LSTM hidden/cell states) that
/// must be carried forward between chunks for accurate detection.
use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::{Tensor, ValueType};

/// Silero VAD model URL (v5, ONNX format).
pub const SILERO_VAD_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx";

/// Silero VAD chunk size: 512 samples at 16kHz = 32ms.
pub const VAD_CHUNK_SAMPLES: usize = 512;
/// Silero streaming context size at 16kHz.
pub const VAD_CONTEXT_SAMPLES: usize = 64;

/// Sample rate expected by Silero VAD.
pub const VAD_SAMPLE_RATE: u32 = 16000;

/// Silero VAD model wrapper.
pub struct SileroVad {
    session: Session,
    /// LSTM hidden state [2, 1, 128] (Silero VAD v5)
    state: Vec<f32>,
    /// Hidden state dimension (detected from model)
    state_dim: usize,
    /// Sample rate as i64 for the model input
    sr: i64,
    /// Rolling context prepended to each frame for streaming inference.
    context: Vec<f32>,
    /// Emit actual runtime I/O tensor shapes once for debugging.
    logged_io_shapes: bool,
}

impl SileroVad {
    /// Load Silero VAD from an ONNX file.
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

        // Initial hidden state: zeros [2, 1, state_dim]
        let state = vec![0.0f32; 2 * 1 * state_dim];

        Ok(Self {
            session,
            state,
            state_dim,
            sr: VAD_SAMPLE_RATE as i64,
            context: vec![0.0f32; VAD_CONTEXT_SAMPLES],
            logged_io_shapes: false,
        })
    }

    /// Reset the internal LSTM state.
    /// Call this when starting a new audio stream or after long pauses.
    pub fn reset(&mut self) {
        reset_state_and_context(&mut self.state, self.state_dim, &mut self.context);
    }

    /// Process a single 512-sample chunk and return speech probability.
    ///
    /// The internal state is updated automatically.
    /// Returns a probability in [0, 1] where higher = more likely speech.
    pub fn process_chunk(&mut self, chunk: &[f32]) -> Result<f32> {
        assert!(
            chunk.len() == VAD_CHUNK_SAMPLES,
            "VAD chunk must be exactly {} samples, got {}",
            VAD_CHUNK_SAMPLES,
            chunk.len()
        );

        let model_input = build_model_input(&self.context, chunk);
        let model_input_len = model_input.len();

        // Input tensor: [1, context + chunk_size]
        let input_tensor = Tensor::from_array(([1usize, model_input_len], model_input.clone()))
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // State tensor: [2, 1, state_dim]
        let state_tensor =
            Tensor::from_array(([2usize, 1usize, self.state_dim], self.state.clone()))
                .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Silero expects an int64 scalar sample-rate tensor.
        let sr_tensor =
            Tensor::from_array(((), vec![self.sr])).map_err(|e| anyhow::anyhow!("{e}"))?;

        if !self.logged_io_shapes {
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

        // Output 0: speech probability [1, 1]
        let (_prob_shape, prob_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let speech_prob = prob_data[0];

        // Output 1: updated state [2, 1, 64]
        let (_state_shape, state_data) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if !self.logged_io_shapes {
            eprintln!("[debug] Silero output stateN shape: {:?}", _state_shape);
            self.logged_io_shapes = true;
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

/// Download the Silero VAD model if it doesn't exist.
pub async fn ensure_vad_model(model_dir: &std::path::Path) -> Result<std::path::PathBuf> {
    let vad_path = model_dir.join("silero_vad.onnx");

    if vad_path.exists() {
        return Ok(vad_path);
    }

    eprintln!("Downloading Silero VAD model...");

    tokio::fs::create_dir_all(model_dir).await?;

    let client = reqwest::Client::builder()
        .user_agent("parakeet-cli/0.1.0")
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

    // Atomic write
    let tmp_path = model_dir.join(".silero_vad.onnx.tmp");
    tokio::fs::write(&tmp_path, &bytes).await?;
    tokio::fs::rename(&tmp_path, &vad_path).await?;

    eprintln!(
        "Silero VAD downloaded ({} bytes) to {}",
        bytes.len(),
        vad_path.display()
    );

    Ok(vad_path)
}

/// State machine for VAD-based speech segmentation.
///
/// Tracks speech/silence transitions and determines when a complete
/// utterance has been detected (speech followed by sufficient silence).
pub struct VadSegmenter {
    /// Speech probability threshold
    threshold: f32,
    /// Number of consecutive silence chunks needed to end an utterance
    silence_chunks_needed: usize,
    /// Current state
    state: VadState,
    /// Count of consecutive silence chunks during speech
    silence_count: usize,
    /// Minimum speech chunks to consider valid (filters noise bursts)
    min_speech_chunks: usize,
    /// Count of speech chunks in current utterance
    speech_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VadState {
    /// Waiting for speech to begin.
    Silence,
    /// Speech detected, accumulating audio.
    Speaking,
}

/// Event emitted by the VAD segmenter.
#[derive(Debug)]
pub enum VadEvent {
    /// No state change — continue as-is.
    None,
    /// Speech has started — begin accumulating audio.
    SpeechStart,
    /// Speech has ended — the utterance is complete, transcribe it.
    SpeechEnd,
}

impl VadSegmenter {
    /// Create a new VAD segmenter.
    ///
    /// # Arguments
    /// * `threshold` - Speech probability threshold (0.0 - 1.0)
    /// * `silence_ms` - Silence duration in ms to end an utterance
    pub fn new(threshold: f32, silence_ms: u64) -> Self {
        // Convert silence duration to number of VAD chunks
        // Each chunk is 512 samples at 16kHz = 32ms
        let chunk_ms = (VAD_CHUNK_SAMPLES as f64 / VAD_SAMPLE_RATE as f64 * 1000.0) as u64;
        let silence_chunks_needed = (silence_ms / chunk_ms).max(1) as usize;

        // Minimum ~100ms of speech to be considered valid
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

    /// Process a speech probability and return the resulting event.
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
                            // Too short, was probably noise
                            VadEvent::None
                        }
                    } else {
                        VadEvent::None
                    }
                }
            }
        }
    }

    /// Reset the segmenter state.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.state = VadState::Silence;
        self.silence_count = 0;
        self.speech_count = 0;
    }

    /// Get the current state.
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

        // Silence
        assert!(matches!(seg.process(0.1), VadEvent::None));
        assert_eq!(seg.state(), VadState::Silence);

        // Speech starts
        assert!(matches!(seg.process(0.8), VadEvent::SpeechStart));
        assert_eq!(seg.state(), VadState::Speaking);

        // Continue speaking for enough chunks to be valid
        for _ in 0..10 {
            assert!(matches!(seg.process(0.9), VadEvent::None));
        }

        // Silence begins but not long enough
        for _ in 0..5 {
            assert!(matches!(seg.process(0.1), VadEvent::None));
            assert_eq!(seg.state(), VadState::Speaking);
        }

        // Enough silence to trigger end (500ms / 32ms = ~16 chunks)
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
