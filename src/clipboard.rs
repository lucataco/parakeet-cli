use anyhow::{Context, Result};
use std::io::Write;
use std::process::{Command, Stdio};

/// Copy text to the macOS clipboard via pbcopy.
pub fn copy_text(text: &str) -> Result<()> {
    let mut child = Command::new("pbcopy")
        .stdin(Stdio::piped())
        .spawn()
        .context("Failed to spawn pbcopy (macOS only)")?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(text.as_bytes())?;
    }

    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("pbcopy exited with status {status}");
    }

    Ok(())
}
