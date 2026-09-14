use anyhow::{Context, Result};
use std::path::Path;

pub struct Tokenizer {
    vocab: Vec<String>,
    pub blank_id: usize,
}

impl Tokenizer {
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

            let (token, id_str) = line
                .rsplit_once(' ')
                .with_context(|| format!("Invalid vocab line: {line}"))?;

            let id: usize = id_str
                .parse()
                .with_context(|| format!("Invalid token ID in line: {line}"))?;

            if id >= vocab.len() {
                vocab.resize(id + 1, String::new());
            }

            vocab[id] = token.to_string();
            if id > max_id {
                max_id = id;
            }
        }

        let blank_id = max_id;

        if verbose {
            eprintln!(
                "Loaded tokenizer: {} tokens, blank_id={}",
                vocab.len(),
                blank_id
            );
        }

        Ok(Self { vocab, blank_id })
    }

    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut text = String::new();

        for &id in token_ids {
            if id >= self.vocab.len() || id == self.blank_id {
                continue;
            }

            let token = &self.vocab[id];
            let piece = token.replace('▁', " ");
            text.push_str(&piece);
        }

        text.trim_start().to_string()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_basic() {
        let tokenizer = Tokenizer {
            vocab: vec![
                "<unk>".to_string(),
                "▁t".to_string(),
                "▁th".to_string(),
                "▁a".to_string(),
                "in".to_string(),
                "▁the".to_string(),
                "<blk>".to_string(),
            ],
            blank_id: 6,
        };

        let result = tokenizer.decode(&[5, 3, 4]);
        assert_eq!(result, "the ain");
    }

    #[test]
    fn test_decode_skips_blank() {
        let tokenizer = Tokenizer {
            vocab: vec![
                "▁Hello".to_string(),
                "▁world".to_string(),
                "<blk>".to_string(),
            ],
            blank_id: 2,
        };

        let result = tokenizer.decode(&[0, 2, 1]);
        assert_eq!(result, "Hello world");
    }
}
