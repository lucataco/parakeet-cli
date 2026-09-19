use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "parakeet",
    version,
    about = "Local speech-to-text powered by NVIDIA Parakeet TDT",
    long_about = "A fast, local speech-to-text CLI using NVIDIA's Parakeet TDT 0.6B v3 model.\n\
                  Supports 25 languages. Runs entirely on-device via ONNX Runtime.\n\
                  INT8 quantized by default for smaller downloads on Apple Silicon."
)]
pub struct Cli {
    #[arg(help = "Enable verbose output (model details, tensor shapes, timing stats)")]
    #[arg(long, short, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    #[command(about = "Print the machine-readable daemon protocol version")]
    ProtocolVersion,
    #[command(about = "Download model weights from HuggingFace")]
    Download {
        #[arg(help = "Directory to store model files")]
        #[arg(long, default_value_os_t = default_model_dir())]
        model_dir: PathBuf,

        #[arg(help = "Download INT8 quantized model (652 MB, smallest). This is the default")]
        #[arg(long, hide = true)]
        int8: bool,

        #[arg(help = "Download FP16 quantized model (1.2 GB) instead of the default INT8 weights")]
        #[arg(long, conflicts_with = "int8")]
        fp16: bool,

        #[arg(
            help = "Progress output: \"auto\" for human-readable bars, \"json\" for machine-readable newline-delimited JSON events on stdout"
        )]
        #[arg(long, default_value = "auto", value_parser = ["auto", "json"])]
        progress: String,
    },

    #[command(about = "Transcribe an audio file")]
    Transcribe {
        #[arg(help = "Replay through the same bounded session pipeline as microphone capture")]
        #[arg(long)]
        session: bool,

        #[arg(
            help = "With --session: also print the interim `partial` NDJSON events the daemon would stream, before the final result"
        )]
        #[arg(long, requires = "session")]
        partials: bool,
        #[arg(help = "Path to the audio file (WAV)")]
        file: PathBuf,

        #[arg(help = "Directory containing model files")]
        #[arg(long, default_value_os_t = default_model_dir())]
        model_dir: PathBuf,

        #[arg(help = "Output format")]
        #[arg(long, default_value = "text", value_parser = ["text", "json"])]
        format: String,

        #[arg(help = "Enable CoreML acceleration (experimental, may be slower with FP32 models)")]
        #[arg(long)]
        coreml: bool,
    },

    #[command(about = "Stream transcription from microphone")]
    Listen {
        #[arg(help = "Audio input device name (use 'devices' command to list)")]
        #[arg(long)]
        device: Option<String>,

        #[arg(help = "Directory containing model files")]
        #[arg(long, default_value_os_t = default_model_dir())]
        model_dir: PathBuf,

        #[arg(help = "VAD speech probability threshold (0.0 - 1.0)")]
        #[arg(long, default_value = "0.5", value_parser = parse_vad_threshold)]
        vad_threshold: f32,

        #[arg(help = "Silence duration in ms to end an utterance")]
        #[arg(long, default_value = "1500")]
        silence_ms: u64,

        #[arg(help = "Also copy transcription to clipboard")]
        #[arg(long)]
        clipboard: bool,

        #[arg(help = "Print debug info: audio levels, VAD probabilities, state transitions")]
        #[arg(long)]
        debug: bool,

        #[arg(help = "Enable CoreML acceleration (experimental, may be slower with FP32 models)")]
        #[arg(long)]
        coreml: bool,

        #[arg(help = "Capture a single utterance and exit (useful for scripting/voice agents)")]
        #[arg(long)]
        single_utterance: bool,
    },

    #[command(about = "Run as a daemon controllable via Unix socket or signals")]
    Serve {
        #[arg(help = "Path to the Unix socket")]
        #[arg(long, default_value_os_t = default_socket_path())]
        socket: PathBuf,

        #[arg(help = "Path to write PID file")]
        #[arg(long, default_value_os_t = default_pid_file_path())]
        pid_file: PathBuf,

        #[arg(help = "Audio input device name")]
        #[arg(long)]
        device: Option<String>,

        #[arg(help = "Directory containing model files")]
        #[arg(long, default_value_os_t = default_model_dir())]
        model_dir: PathBuf,

        #[arg(help = "Copy transcription to clipboard")]
        #[arg(long)]
        clipboard: bool,

        #[arg(help = "Enable CoreML acceleration (experimental, may be slower with FP32 models)")]
        #[arg(long)]
        coreml: bool,
    },

    #[command(about = "List available audio input devices")]
    Devices,
}

fn default_model_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("parakeet")
        .join("models")
        .join("parakeet-tdt-0.6b-v3")
}

fn default_runtime_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("parakeet")
        .join("run")
}

fn default_socket_path() -> PathBuf {
    default_runtime_dir().join("daemon.sock")
}

fn default_pid_file_path() -> PathBuf {
    default_runtime_dir().join("daemon.pid")
}

fn parse_vad_threshold(value: &str) -> Result<f32, String> {
    let threshold: f32 = value
        .parse()
        .map_err(|_| format!("invalid VAD threshold '{value}'"))?;

    if (0.0..=1.0).contains(&threshold) {
        Ok(threshold)
    } else {
        Err(format!(
            "VAD threshold must be between 0.0 and 1.0, got {threshold}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_dir_uses_v3_path() {
        let model_dir = default_model_dir();

        assert!(model_dir.ends_with("parakeet/models/parakeet-tdt-0.6b-v3"));
    }

    #[test]
    fn download_defaults_to_int8() {
        let cli = Cli::try_parse_from(["parakeet", "download"]).expect("download parses");

        match cli.command {
            Commands::Download { int8, fp16, .. } => {
                assert!(!int8);
                assert!(!fp16);
            }
            _ => panic!("expected download command"),
        }
    }

    #[test]
    fn download_accepts_fp16_override() {
        let cli = Cli::try_parse_from(["parakeet", "download", "--fp16"])
            .expect("download with fp16 parses");

        match cli.command {
            Commands::Download { fp16, .. } => assert!(fp16),
            _ => panic!("expected download command"),
        }
    }

    #[test]
    fn download_progress_defaults_to_auto() {
        let cli = Cli::try_parse_from(["parakeet", "download"]).expect("download parses");

        match cli.command {
            Commands::Download { progress, .. } => assert_eq!(progress, "auto"),
            _ => panic!("expected download command"),
        }
    }

    #[test]
    fn download_accepts_json_progress() {
        let cli = Cli::try_parse_from(["parakeet", "download", "--progress", "json"])
            .expect("download with json progress parses");

        match cli.command {
            Commands::Download { progress, .. } => assert_eq!(progress, "json"),
            _ => panic!("expected download command"),
        }
    }

    #[test]
    fn download_rejects_unknown_progress() {
        assert!(Cli::try_parse_from(["parakeet", "download", "--progress", "xml"]).is_err());
    }

    #[test]
    fn vad_threshold_parser_accepts_bounds() {
        assert_eq!(parse_vad_threshold("0").unwrap(), 0.0);
        assert_eq!(parse_vad_threshold("1.0").unwrap(), 1.0);
    }

    #[test]
    fn vad_threshold_parser_rejects_out_of_range_values() {
        assert!(parse_vad_threshold("-0.1").is_err());
        assert!(parse_vad_threshold("1.1").is_err());
    }
}
