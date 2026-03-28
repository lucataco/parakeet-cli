use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "parakeet",
    version,
    about = "Local speech-to-text powered by NVIDIA Parakeet TDT",
    long_about = "A fast, local speech-to-text CLI using NVIDIA's Parakeet TDT 0.6B v3 model.\n\
                  Supports 25 languages. Runs entirely on-device via ONNX Runtime.\n\
                  FP16 quantized by default for optimal speed on Apple Silicon."
)]
pub struct Cli {
    /// Enable verbose output (model details, tensor shapes, timing stats)
    #[arg(long, short, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Download model weights from HuggingFace
    Download {
        /// Directory to store model files
        #[arg(long, default_value_os_t = default_model_dir())]
        model_dir: PathBuf,

        /// Download INT8 quantized model (652 MB, smallest). Default is FP16 (1.2 GB).
        #[arg(long)]
        int8: bool,
    },

    /// Transcribe an audio file
    Transcribe {
        /// Path to the audio file (WAV or FLAC)
        file: PathBuf,

        /// Directory containing model files
        #[arg(long, default_value_os_t = default_model_dir())]
        model_dir: PathBuf,

        /// Include word-level timestamps
        #[arg(long)]
        timestamps: bool,

        /// Output format
        #[arg(long, default_value = "text", value_parser = ["text", "json", "srt"])]
        format: String,

        /// Enable CoreML acceleration (experimental, may be slower with FP32 models)
        #[arg(long)]
        coreml: bool,
    },

    /// Stream transcription from microphone
    Listen {
        /// Audio input device name (use 'devices' command to list)
        #[arg(long)]
        device: Option<String>,

        /// Directory containing model files
        #[arg(long, default_value_os_t = default_model_dir())]
        model_dir: PathBuf,

        /// VAD speech probability threshold (0.0 - 1.0)
        #[arg(long, default_value = "0.5")]
        vad_threshold: f32,

        /// Silence duration in ms to end an utterance
        #[arg(long, default_value = "1500")]
        silence_ms: u64,

        /// Copy transcription to clipboard instead of stdout
        #[arg(long)]
        clipboard: bool,

        /// Print debug info: audio levels, VAD probabilities, state transitions
        #[arg(long)]
        debug: bool,

        /// Enable CoreML acceleration (experimental, may be slower with FP32 models)
        #[arg(long)]
        coreml: bool,
    },

    /// Run as a daemon controllable via Unix socket or signals
    Serve {
        /// Path to the Unix socket
        #[arg(long, default_value = "/tmp/parakeet.sock")]
        socket: PathBuf,

        /// Path to write PID file
        #[arg(long, default_value = "/tmp/parakeet.pid")]
        pid_file: PathBuf,

        /// Audio input device name
        #[arg(long)]
        device: Option<String>,

        /// Directory containing model files
        #[arg(long, default_value_os_t = default_model_dir())]
        model_dir: PathBuf,

        /// Copy transcription to clipboard
        #[arg(long)]
        clipboard: bool,

        /// Enable CoreML acceleration (experimental, may be slower with FP32 models)
        #[arg(long)]
        coreml: bool,
    },

    /// List available audio input devices
    Devices,
}

fn default_model_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("parakeet")
        .join("models")
        .join("parakeet-tdt-0.6b-v3")
}
