use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

const HF_BASE_URL: &str = "https://huggingface.co";

/// HuggingFace repo for FP16 quantized model files (v3).
const HF_REPO_FP16: &str = "grikdotnet/parakeet-tdt-0.6b-fp16";

/// HuggingFace repo for FP32/INT8 model files + vocab + config (v3).
const HF_REPO_V3: &str = "istupakov/parakeet-tdt-0.6b-v3-onnx";

/// Files needed for the FP16 model (default).
/// Encoder + decoder come from the FP16 repo; vocab + config from the v3 repo.
struct DownloadFile {
    repo: &'static str,
    filename: &'static str,
}

const FP16_FILES: &[DownloadFile] = &[
    DownloadFile {
        repo: HF_REPO_FP16,
        filename: "encoder-model.fp16.onnx",
    },
    DownloadFile {
        repo: HF_REPO_FP16,
        filename: "decoder_joint-model.fp16.onnx",
    },
    DownloadFile {
        repo: HF_REPO_V3,
        filename: "vocab.txt",
    },
    DownloadFile {
        repo: HF_REPO_V3,
        filename: "config.json",
    },
];

/// Files needed for the INT8 quantized model.
const INT8_FILES: &[DownloadFile] = &[
    DownloadFile {
        repo: HF_REPO_V3,
        filename: "encoder-model.int8.onnx",
    },
    DownloadFile {
        repo: HF_REPO_V3,
        filename: "decoder_joint-model.int8.onnx",
    },
    DownloadFile {
        repo: HF_REPO_V3,
        filename: "vocab.txt",
    },
    DownloadFile {
        repo: HF_REPO_V3,
        filename: "config.json",
    },
];

pub async fn download_model(model_dir: &Path, int8: bool) -> Result<()> {
    let files = if int8 { INT8_FILES } else { FP16_FILES };
    let variant = if int8 { "INT8 quantized" } else { "FP16" };

    println!("Downloading Parakeet TDT 0.6B v3 ({variant}) model...");
    if int8 {
        println!("Source: {HF_BASE_URL}/{HF_REPO_V3}");
    } else {
        println!("Source: {HF_BASE_URL}/{HF_REPO_FP16}");
    }
    println!("Destination: {}", model_dir.display());
    println!();

    // Create model directory
    fs::create_dir_all(model_dir)
        .await
        .with_context(|| format!("Failed to create directory: {}", model_dir.display()))?;

    let client = Client::builder().user_agent("parakeet-cli/0.1.0").build()?;

    for dl in files {
        let dest_path = model_dir.join(dl.filename);

        if dest_path.exists() {
            println!("[skip] {} (already exists)", dl.filename);
            continue;
        }

        download_file(&client, dl.repo, dl.filename, &dest_path).await?;
    }

    // Write a marker file indicating which variant is downloaded
    let marker = model_dir.join(".variant");
    fs::write(&marker, variant).await?;

    println!();
    println!("Download complete! Model ready at: {}", model_dir.display());

    Ok(())
}

async fn download_file(
    client: &Client,
    repo: &str,
    filename: &str,
    dest_path: &Path,
) -> Result<()> {
    let url = format!("{HF_BASE_URL}/{repo}/resolve/main/{filename}");

    let response = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("Failed to request {filename}"))?;

    if !response.status().is_success() {
        anyhow::bail!("Failed to download {filename}: HTTP {}", response.status());
    }

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta}) {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message(filename.to_string());

    // Create a temporary file for atomic writes
    let tmp_path = temp_path(dest_path);
    let mut file = fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("Failed to create file: {}", tmp_path.display()))?;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("Error downloading {filename}"))?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }

    file.flush().await?;
    drop(file);

    // Atomic rename
    fs::rename(&tmp_path, dest_path).await.with_context(|| {
        format!(
            "Failed to rename {} -> {}",
            tmp_path.display(),
            dest_path.display()
        )
    })?;

    pb.finish_with_message(format!("{filename} done"));

    Ok(())
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
