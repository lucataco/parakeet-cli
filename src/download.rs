use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

const HF_BASE_URL: &str = "https://huggingface.co";
const USER_AGENT: &str = concat!("parakeet-cli/", env!("CARGO_PKG_VERSION"));

/// HuggingFace repo for FP16 quantized model files (v3).
const HF_REPO_FP16: &str = "grikdotnet/parakeet-tdt-0.6b-fp16";
const HF_REPO_FP16_REVISION: &str = "dc9871ec5ad84a420940077e76e8741b3609bf8b";

/// HuggingFace repo for INT8 model files + vocab + config (v3).
const HF_REPO_V3: &str = "istupakov/parakeet-tdt-0.6b-v3-onnx";
const HF_REPO_V3_REVISION: &str = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce";

/// Files needed for the FP16 model (default).
/// Encoder + decoder come from the FP16 repo; vocab + config from the v3 repo.
struct DownloadFile {
    repo: &'static str,
    revision: &'static str,
    filename: &'static str,
    sha256: &'static str,
}

const FP16_FILES: &[DownloadFile] = &[
    DownloadFile {
        repo: HF_REPO_FP16,
        revision: HF_REPO_FP16_REVISION,
        filename: "encoder-model.fp16.onnx",
        sha256: "a2bdeeb99cb7e5548818e823127b33854dd0c26f5d0c8da91effdd895ea0e717",
    },
    DownloadFile {
        repo: HF_REPO_FP16,
        revision: HF_REPO_FP16_REVISION,
        filename: "decoder_joint-model.fp16.onnx",
        sha256: "b33a73b7c1d71b9d5a0911f5cb478be3dcbf79f53355c531ab1cd1dcd68ad8ef",
    },
    DownloadFile {
        repo: HF_REPO_V3,
        revision: HF_REPO_V3_REVISION,
        filename: "vocab.txt",
        sha256: "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
    },
    DownloadFile {
        repo: HF_REPO_V3,
        revision: HF_REPO_V3_REVISION,
        filename: "config.json",
        sha256: "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
    },
];

/// Files needed for the INT8 quantized model.
const INT8_FILES: &[DownloadFile] = &[
    DownloadFile {
        repo: HF_REPO_V3,
        revision: HF_REPO_V3_REVISION,
        filename: "encoder-model.int8.onnx",
        sha256: "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09",
    },
    DownloadFile {
        repo: HF_REPO_V3,
        revision: HF_REPO_V3_REVISION,
        filename: "decoder_joint-model.int8.onnx",
        sha256: "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
    },
    DownloadFile {
        repo: HF_REPO_V3,
        revision: HF_REPO_V3_REVISION,
        filename: "vocab.txt",
        sha256: "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
    },
    DownloadFile {
        repo: HF_REPO_V3,
        revision: HF_REPO_V3_REVISION,
        filename: "config.json",
        sha256: "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
    },
];

pub async fn download_model(model_dir: &Path, int8: bool) -> Result<()> {
    let files = if int8 { INT8_FILES } else { FP16_FILES };
    let variant = if int8 { "INT8 quantized" } else { "FP16" };

    println!("Downloading Parakeet TDT 0.6B v3 ({variant}) model...");
    println!("Destination: {}", model_dir.display());
    println!();

    fs::create_dir_all(model_dir)
        .await
        .with_context(|| format!("Failed to create directory: {}", model_dir.display()))?;

    let client = Client::builder()
        .https_only(true)
        .user_agent(USER_AGENT)
        .build()?;

    for dl in files {
        let dest_path = model_dir.join(dl.filename);

        if dest_path.exists() {
            if file_matches_sha256(&dest_path, dl.sha256)? {
                println!("[skip] {} (already verified)", dl.filename);
                continue;
            }

            println!("[redownload] {} (checksum mismatch)", dl.filename);
            fs::remove_file(&dest_path).await.with_context(|| {
                format!(
                    "Failed to remove invalid cached file before re-download: {}",
                    dest_path.display()
                )
            })?;
        }

        download_file(&client, dl, &dest_path).await?;
    }

    let marker = model_dir.join(".variant");
    fs::write(&marker, variant).await?;

    println!();
    println!("Download complete! Model ready at: {}", model_dir.display());

    Ok(())
}

async fn download_file(client: &Client, spec: &DownloadFile, dest_path: &Path) -> Result<()> {
    let url = format!(
        "{HF_BASE_URL}/{}/resolve/{}/{}",
        spec.repo, spec.revision, spec.filename
    );

    let response = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("Failed to request {}", spec.filename))?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Failed to download {}: HTTP {}",
            spec.filename,
            response.status()
        );
    }

    let total_size = response.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}")
            .context("Failed to build download progress bar template")?
            .progress_chars("=>-"),
    );
    pb.set_message(spec.filename.to_string());

    let tmp_path = temp_path(dest_path);
    let _ = fs::remove_file(&tmp_path).await;
    let mut file = fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("Failed to create file: {}", tmp_path.display()))?;

    let mut hasher = Sha256::new();
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("Error downloading {}", spec.filename))?;
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }

    file.flush().await?;
    file.sync_all().await?;
    drop(file);

    let actual_sha256 = format!("{:x}", hasher.finalize());
    if actual_sha256 != spec.sha256 {
        let _ = fs::remove_file(&tmp_path).await;
        anyhow::bail!(
            "Checksum mismatch for {}: expected {}, got {}",
            spec.filename,
            spec.sha256,
            actual_sha256,
        );
    }

    fs::rename(&tmp_path, dest_path).await.with_context(|| {
        format!(
            "Failed to rename {} -> {}",
            tmp_path.display(),
            dest_path.display()
        )
    })?;

    pb.finish_with_message(format!("{} verified", spec.filename));

    Ok(())
}

fn file_matches_sha256(path: &Path, expected_sha256: &str) -> Result<bool> {
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

fn temp_path(path: &Path) -> PathBuf {
    let mut tmp = path.to_path_buf();
    let name = tmp
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    tmp.set_file_name(format!(".{name}.tmp"));
    tmp
}

/// Check if model files exist at the given directory.
///
/// Checks for FP16, INT8, or legacy FP32 files (in priority order).
pub fn model_exists(model_dir: &Path) -> bool {
    let has_fp16 = model_dir.join("encoder-model.fp16.onnx").exists()
        && model_dir.join("decoder_joint-model.fp16.onnx").exists();
    let has_int8 = model_dir.join("encoder-model.int8.onnx").exists()
        && model_dir.join("decoder_joint-model.int8.onnx").exists();
    let has_fp32 = model_dir.join("encoder-model.onnx").exists()
        && model_dir.join("decoder_joint-model.onnx").exists();

    (has_fp16 || has_int8 || has_fp32)
        && model_dir.join("vocab.txt").exists()
        && model_dir.join("config.json").exists()
}
