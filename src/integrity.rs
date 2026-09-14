use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

pub(crate) fn file_matches_sha256(path: &Path, expected_sha256: &str) -> Result<bool> {
    let file = std::fs::File::open(path).with_context(|| {
        format!(
            "Failed to open file for checksum verification: {}",
            path.display()
        )
    })?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buf).with_context(|| {
            format!(
                "Failed to read file for checksum verification: {}",
                path.display()
            )
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()) == expected_sha256)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_match_mismatch_and_missing_file() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("parakeet-integrity-{}-{stamp}", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        let matched = file_matches_sha256(
            &path,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
        let mismatched = file_matches_sha256(&path, "incorrect");
        std::fs::remove_file(&path).unwrap();
        assert!(matched.unwrap());
        assert!(!mismatched.unwrap());
        assert!(
            file_matches_sha256(&path, "incorrect")
                .unwrap_err()
                .to_string()
                .contains("Failed to open file for checksum verification")
        );
    }
}
