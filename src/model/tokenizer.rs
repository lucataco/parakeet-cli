use anyhow::{Context, Result};
use std::path::Path;

/// Tokenizer for Parakeet TDT models.
///
/// Loads the SentencePiece unigram vocabulary from vocab.txt.
/// Handles decoding of token IDs back to text, including the
/// SentencePiece `▁` (U+2581) word boundary marker.
pub struct Tokenizer {
    /// Token ID -> token string
    vocab: Vec<String>,
    /// The blank token ID (last token in vocab)
    pub blank_id: usize,
}

impl Tokenizer {
    /// Load tokenizer from a vocab.txt file.
    ///
    /// Expected format: `<token> <id>` per line.
    /// The last entry should be `<blk>` (blank token).
    pub fn from_file(path: &Path, verbose: bool) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read vocab file: {}", path.display()))?;

        let mut vocab = Vec::new();
        let mut max_id = 0usize;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Format: "token id" — split from the right since tokens can contain spaces
            let (token, id_str) = line
                .rsplit_once(' ')
                .with_context(|| format!("Invalid vocab line: {line}"))?;

            let id: usize = id_str
                .parse()
                .with_context(|| format!("Invalid token ID in line: {line}"))?;

            // Ensure vocab is large enough
            if id >= vocab.len() {
                vocab.resize(id + 1, String::new());
            }

            vocab[id] = token.to_string();
            if id > max_id {
                max_id = id;
            }
        }

        // The blank token should be the last one (<blk>)
        let blank_id = max_id;

        if verbose {
            println!(
                "Loaded tokenizer: {} tokens, blank_id={}",
                vocab.len(),
                blank_id
            );
        }

        Ok(Self { vocab, blank_id })
    }

    /// Decode a sequence of token IDs into text.
    ///
    /// Handles SentencePiece `▁` markers by converting them to spaces,
    /// and strips leading whitespace from the result.
    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut text = String::new();

        for &id in token_ids {
            if id >= self.vocab.len() || id == self.blank_id {
                continue;
            }

            let token = &self.vocab[id];
            // SentencePiece uses ▁ (U+2581) as word boundary
            let piece = token.replace('▁', " ");
            text.push_str(&piece);
        }

        // Strip leading space that comes from the first ▁
        text.trim_start().to_string()
    }

    /// Get the vocabulary size (including blank token)
    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Look up a token by ID
    #[allow(dead_code)]
    pub fn token(&self, id: usize) -> Option<&str> {
        self.vocab.get(id).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_basic() {
        let tokenizer = Tokenizer {
            vocab: vec![
                "<unk>".to_string(), // 0
                "▁t".to_string(),    // 1
                "▁th".to_string(),   // 2
                "▁a".to_string(),    // 3
                "in".to_string(),    // 4
                "▁the".to_string(),  // 5
                "<blk>".to_string(), // 6
            ],
            blank_id: 6,
        };

        // "the" + "a" + "in" -> "the a in"  (with SentencePiece boundaries)
        let result = tokenizer.decode(&[5, 3, 4]);
        // ▁the -> " the", ▁a -> " a", in -> "in"
        // Full: " the a in" -> trimmed: "the ain"
        // Wait: "▁the" = " the", "▁a" = " a", "in" = "in" => " the a in" => "the ain"
        // Hmm, that's "the ain" not "the a in" — let me reconsider
        // Actually: decode [5, 3, 4] = "▁the" + "▁a" + "in" = " the" + " a" + "in" = " the ain"
        // trimmed = "the ain"
        assert_eq!(result, "the ain");
    }

    #[test]
    fn test_decode_skips_blank() {
        let tokenizer = Tokenizer {
            vocab: vec![
                "▁Hello".to_string(), // 0
                "▁world".to_string(), // 1
                "<blk>".to_string(),  // 2
            ],
            blank_id: 2,
        };

        let result = tokenizer.decode(&[0, 2, 1]);
        assert_eq!(result, "Hello world");
    }
}
