use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

const HF_REPO: &str = "istupakov/parakeet-tdt-0.6b-v2-onnx";
const HF_BASE_URL: &str = "https://huggingface.co";

/// Files needed for the FP32 model
const FP32_FILES: &[&str] = &[
    "encoder-model.onnx",
    "encoder-model.onnx.data",
    "decoder_joint-model.onnx",
    "vocab.txt",
    "config.json",
];

/// Files needed for the INT8 quantized model
const INT8_FILES: &[&str] = &[
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "vocab.txt",
    "config.json",
];

pub async fn download_model(model_dir: &Path, int8: bool) -> Result<()> {
    let files = if int8 { INT8_FILES } else { FP32_FILES };
    let variant = if int8 { "INT8 quantized" } else { "FP32" };

    println!("Downloading Parakeet TDT 0.6B v2 ({variant}) model...");
    println!("Source: {HF_BASE_URL}/{HF_REPO}");
    println!("Destination: {}", model_dir.display());
    println!();

    // Create model directory
    fs::create_dir_all(model_dir)
        .await
        .with_context(|| format!("Failed to create directory: {}", model_dir.display()))?;

    let client = Client::builder().user_agent("parakeet-cli/0.1.0").build()?;

    for filename in files {
        let dest_path = model_dir.join(filename);

        if dest_path.exists() {
            println!("[skip] {filename} (already exists)");
            continue;
        }

        download_file(&client, filename, &dest_path).await?;
    }

    // Write a marker file indicating which variant is downloaded
    let marker = model_dir.join(".variant");
    fs::write(&marker, variant).await?;

    println!();
    println!("Download complete! Model ready at: {}", model_dir.display());

    Ok(())
}

async fn download_file(client: &Client, filename: &str, dest_path: &Path) -> Result<()> {
    let url = format!("{HF_BASE_URL}/{HF_REPO}/resolve/main/{filename}");

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

/// Check if the model files exist at the given directory
pub fn model_exists(model_dir: &Path, int8: bool) -> bool {
    let files = if int8 { INT8_FILES } else { FP32_FILES };
    files.iter().all(|f| model_dir.join(f).exists())
}
