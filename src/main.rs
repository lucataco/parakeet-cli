use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};
use parakeet_cli::{audio, cli, download, listen, model, serve};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let verbose = cli.verbose;

    match cli.command {
        Commands::Download {
            model_dir,
            int8: _,
            fp16,
        } => {
            download::download_model(&model_dir, !fp16).await?;
        }

        Commands::Transcribe {
            file,
            model_dir,
            format,
            coreml,
        } => {
            // Verify model exists
            if !download::model_exists(&model_dir) {
                eprintln!(
                    "Model not found at: {}\nRun `parakeet download` first.",
                    model_dir.display()
                );
                std::process::exit(1);
            }

            // Verify input file exists
            if !file.exists() {
                eprintln!("Audio file not found: {}", file.display());
                std::process::exit(1);
            }

            // Load audio file
            let start_load = std::time::Instant::now();
            let samples = audio::load_wav_file(&file, verbose)?;
            if verbose {
                println!("Audio loaded in {:.2}s", start_load.elapsed().as_secs_f64());
                println!();
            }

            // Compute mel spectrogram
            let start_mel = std::time::Instant::now();
            let mel_config = audio::MelConfig::default();
            let features = audio::compute_mel_spectrogram(&samples, &mel_config);
            if verbose {
                println!(
                    "Mel spectrogram: {} frames x {} bins ({:.2}s)",
                    features.shape()[0],
                    features.shape()[1],
                    start_mel.elapsed().as_secs_f64()
                );
                println!();
            }

            // Load model
            let start_model = std::time::Instant::now();
            let mut model = model::ParakeetModel::load(&model_dir, coreml, verbose)?;
            if verbose {
                println!(
                    "Model loaded in {:.2}s",
                    start_model.elapsed().as_secs_f64()
                );
                println!();
            }

            // Transcribe
            let start_infer = std::time::Instant::now();
            let text = model.transcribe(&features)?;
            let infer_time = start_infer.elapsed().as_secs_f64();
            let audio_duration = samples.len() as f64 / 16000.0;

            match format.as_str() {
                "json" => {
                    let output = serde_json::json!({
                        "text": text,
                        "duration": audio_duration,
                        "inference_time": infer_time,
                    });
                    println!("{}", serde_json::to_string_pretty(&output)?);
                }
                _ => {
                    // "text" format (default)
                    println!("{text}");
                }
            }

            if verbose {
                eprintln!();
                eprintln!(
                    "Transcribed {:.1}s of audio in {:.2}s ({:.1}x realtime)",
                    audio_duration,
                    infer_time,
                    audio_duration / infer_time
                );
            }
        }

        Commands::Listen {
            device,
            model_dir,
            vad_threshold,
            silence_ms,
            clipboard,
            debug,
            coreml,
            single_utterance,
        } => {
            // Verify model exists
            if !download::model_exists(&model_dir) {
                eprintln!(
                    "Model not found at: {}\nRun `parakeet download` first.",
                    model_dir.display()
                );
                std::process::exit(1);
            }

            listen::run_listen(
                &device,
                &model_dir,
                vad_threshold,
                silence_ms,
                clipboard,
                debug,
                verbose,
                coreml,
                single_utterance,
            )
            .await?;
        }

        Commands::Serve {
            socket,
            pid_file,
            device,
            model_dir,
            clipboard,
            coreml,
        } => {
            // Verify model exists
            if !download::model_exists(&model_dir) {
                eprintln!(
                    "Model not found at: {}\nRun `parakeet download` first.",
                    model_dir.display()
                );
                std::process::exit(1);
            }

            serve::run_serve(
                &socket, &pid_file, &device, &model_dir, clipboard, verbose, coreml,
            )
            .await?;
        }

        Commands::Devices => {
            audio::print_input_devices()?;
        }
    }

    Ok(())
}
