use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

/// TDT (Token-and-Duration Transducer) decoder for Parakeet.
///
/// The decoder has a prediction network (LSTM-based) that maintains
/// hidden states across decoding steps. The joint network combines
/// encoder and prediction outputs to produce token + duration logits.
///
/// Output shape is [B, T, U, vocab_size + num_durations].
/// For v3 (multilingual): 8193 + 5 = 8198
pub struct TdtDecoder {
    session: Session,
    /// Vocab size (including blank token)
    vocab_size: usize,
    /// Number of duration classes (5: 0, 1, 2, 3, 4)
    num_durations: usize,
    /// LSTM hidden dimension
    lstm_hidden: usize,
}

impl TdtDecoder {
    /// Load the decoder_joint ONNX model.
    ///
    /// `vocab_size` should match the tokenizer's vocab size (including
    /// the blank token). This is used to split the decoder output into
    /// token logits and duration logits.
    ///
    /// CPU only -- the decoder is small and autoregressive.
    pub fn load(path: &Path, vocab_size: usize, verbose: bool) -> Result<Self> {
        let mut builder = Session::builder().map_err(|e| anyhow::anyhow!("{e}"))?;
        let session = builder
            .commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("Failed to load decoder model: {}", path.display()))?;

        // Log model info
        if verbose {
            println!("Decoder loaded (vocab_size={vocab_size}):");
            for input in session.inputs() {
                println!("  input: {} {:?}", input.name(), input.dtype());
            }
            for output in session.outputs() {
                println!("  output: {} {:?}", output.name(), output.dtype());
            }
        }

        Ok(Self {
            session,
            vocab_size,
            num_durations: 5,
            lstm_hidden: 640,
        })
    }

    /// Run greedy TDT decoding on encoder output.
    ///
    /// # Arguments
    /// * `encoder_output` - Flat f32 vec from encoder, shape [batch, hidden_dim, time]
    /// * `enc_shape` - Shape [batch, hidden_dim, time]  (note: transposed layout)
    /// * `encoded_length` - Number of valid encoder frames
    /// * `blank_id` - Token ID for the blank symbol
    ///
    /// # Returns
    /// * Vector of decoded token IDs (excluding blanks)
    pub fn decode_greedy(
        &mut self,
        encoder_output: &[f32],
        enc_shape: &[usize],
        encoded_length: i64,
        blank_id: usize,
    ) -> Result<Vec<usize>> {
        if enc_shape.len() != 3 {
            anyhow::bail!(
                "Encoder returned invalid shape {:?}; expected [batch, hidden_dim, time]",
                enc_shape
            );
        }
        if encoded_length < 0 {
            anyhow::bail!("Encoder returned negative encoded length: {encoded_length}");
        }

        let _batch = enc_shape[0];
        let hidden_dim = enc_shape[1];
        let time_dim = enc_shape[2];

        let expected_len = hidden_dim
            .checked_mul(time_dim)
            .context("Encoder output shape overflowed while validating dimensions")?;
        if encoder_output.len() < expected_len {
            anyhow::bail!(
                "Encoder output buffer too small: got {}, expected at least {} for shape {:?}",
                encoder_output.len(),
                expected_len,
                enc_shape,
            );
        }

        let max_steps = (encoded_length as usize).min(time_dim);

        let mut tokens: Vec<usize> = Vec::new();
        let mut position: usize = 0;

        // Initial decoder state
        let mut last_label: i32 = blank_id as i32;

        // LSTM states: shape [2, 1, 640] — two layers, batch=1
        let mut state1 = vec![0.0f32; 2 * 1 * self.lstm_hidden];
        let mut state2 = vec![0.0f32; 2 * 1 * self.lstm_hidden];

        // Safety limit to prevent infinite loops
        let max_iterations = max_steps * 10;
        let mut iterations = 0;

        while position < max_steps && iterations < max_iterations {
            iterations += 1;

            // Extract encoder output at current position
            // Encoder output is [batch=1, hidden_dim=1024, time] in row-major
            // We need [1, 1024, 1] slice at position
            let mut enc_slice = vec![0.0f32; hidden_dim];
            for h in 0..hidden_dim {
                enc_slice[h] = encoder_output[h * time_dim + position];
            }

            let enc_tensor = Tensor::from_array(([1usize, hidden_dim, 1], enc_slice))
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            // Decoder inputs
            let targets = Tensor::from_array(([1usize, 1], vec![last_label]))
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            let target_length =
                Tensor::from_array(([1usize], vec![1i32])).map_err(|e| anyhow::anyhow!("{e}"))?;

            let states1_tensor =
                Tensor::from_array(([2usize, 1, self.lstm_hidden], state1.clone()))
                    .map_err(|e| anyhow::anyhow!("{e}"))?;

            let states2_tensor =
                Tensor::from_array(([2usize, 1, self.lstm_hidden], state2.clone()))
                    .map_err(|e| anyhow::anyhow!("{e}"))?;

            // Run decoder + joint network
            let outputs = self
                .session
                .run(ort::inputs![
                    "encoder_outputs" => enc_tensor,
                    "targets" => targets,
                    "target_length" => target_length,
                    "input_states_1" => states1_tensor,
                    "input_states_2" => states2_tensor,
                ])
                .map_err(|e| anyhow::anyhow!("{e}"))
                .context("Decoder inference failed")?;

            // Output 0: logits [1, 1, 1, 1030] (vocab + durations combined)
            let (_logits_shape, logits_data) = outputs["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow::anyhow!("{e}"))
                .context("Failed to extract logits")?;

            // Parse combined logits: first 1025 = token logits, last 5 = duration logits
            let total = self.vocab_size + self.num_durations; // 1030
            if logits_data.len() < total {
                anyhow::bail!(
                    "Decoder logits too small: got {}, expected at least {total}",
                    logits_data.len(),
                );
            }
            let offset = logits_data.len() - total;
            let token_logits = &logits_data[offset..offset + self.vocab_size];
            let duration_logits = &logits_data[offset + self.vocab_size..];

            let (token_id, _) = argmax(token_logits);
            let (duration, _) = argmax(duration_logits);

            if token_id == blank_id {
                // Blank: advance position, don't update LSTM states
                position += duration.max(1);
            } else {
                // Non-blank: emit token, update states, advance
                tokens.push(token_id);
                last_label = token_id as i32;

                // Update LSTM states from outputs
                let (_s1_shape, s1_data) = outputs["output_states_1"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                state1 = s1_data.to_vec();

                let (_s2_shape, s2_data) = outputs["output_states_2"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                state2 = s2_data.to_vec();

                position += duration.max(1);
            }
        }

        Ok(tokens)
    }
}

/// Find the index and value of the maximum element in a slice.
fn argmax(slice: &[f32]) -> (usize, f32) {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    (max_idx, max_val)
}
