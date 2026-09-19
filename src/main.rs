use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};
use parakeet_cli::{audio, cli, download, listen, model, serve};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let verbose = cli.verbose;

    match cli.command {
        Commands::ProtocolVersion => println!("{}", parakeet_cli::DAEMON_PROTOCOL_VERSION),
        Commands::Download {
            model_dir,
            int8: _,
            fp16,
            progress,
        } => {
            let mode = if progress == "json" {
                download::ProgressMode::Json
            } else {
                download::ProgressMode::Bar
            };
            download::download_model(&model_dir, !fp16, mode).await?;
        }

        Commands::Transcribe {
            session,
            partials,
            file,
            model_dir,
            format,
            coreml,
        } => {
            if session {
                let result = serve::replay(&file, &model_dir, coreml, partials).await?;
                if format == "json" {
                    println!("{}", result.completion("replay"));
                } else {
                    println!("{}", result.text);
                }
                return Ok(());
            }
            if !download::model_exists(&model_dir) {
                eprintln!(
                    "Model not found at: {}\nRun `parakeet download` first.",
                    model_dir.display()
                );
                std::process::exit(1);
            }

            if !file.exists() {
                eprintln!("Audio file not found: {}", file.display());
                std::process::exit(1);
            }

            let start_load = std::time::Instant::now();
            let samples = audio::load_wav_file(&file, verbose)?;
            if verbose {
                eprintln!("Audio loaded in {:.2}s", start_load.elapsed().as_secs_f64());
                eprintln!();
            }

            let start_mel = std::time::Instant::now();
            let mel_config = audio::MelConfig::default();
            let features = audio::compute_mel_spectrogram(&samples, &mel_config);
            if verbose {
                eprintln!(
                    "Mel spectrogram: {} frames x {} bins ({:.2}s)",
                    features.shape()[0],
                    features.shape()[1],
                    start_mel.elapsed().as_secs_f64()
                );
                eprintln!();
            }

            let start_model = std::time::Instant::now();
            let mut model = model::ParakeetModel::load(&model_dir, coreml, verbose)?;
            if verbose {
                eprintln!(
                    "Model loaded in {:.2}s",
                    start_model.elapsed().as_secs_f64()
                );
                eprintln!();
            }

            let start_infer = std::time::Instant::now();
            let text = model.transcribe(&features)?;
            let infer_time = start_infer.elapsed().as_secs_f64();
            let audio_duration = samples.len() as f64 / audio::TARGET_SAMPLE_RATE as f64;

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
            if !download::model_exists(&model_dir) {
                eprintln!(
                    "Model not found at: {}\nRun `parakeet download` first.",
                    model_dir.display()
                );
                std::process::exit(1);
            }

            listen::run_listen(listen::ListenConfig {
                device: &device,
                model_dir: &model_dir,
                vad_threshold,
                silence_ms,
                clipboard,
                debug,
                verbose,
                use_coreml: coreml,
                single_utterance,
            })
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
