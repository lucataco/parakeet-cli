# parakeet-cli

Local speech-to-text CLI powered by NVIDIA's Parakeet TDT 0.6B v3 model. Runs entirely on-device via ONNX Runtime — no cloud APIs, no network required after model download.

## Features

- **High accuracy** — NVIDIA Parakeet TDT 0.6B v3 with punctuation and capitalization
- **25 languages** — English, Spanish, French, German, Italian, Portuguese, Russian, Polish, Ukrainian, Czech, Slovak, Dutch, Swedish, Danish, Finnish, and more
- **Fully local** — all inference runs on-device; audio never leaves your machine
- **Fast** — ~59x realtime on Apple Silicon CPU with FP16 quantization
- **Small footprint** — 1.2 GB FP16 model (or 652 MB INT8)
- **File transcription** — transcribe WAV files with text or JSON output
- **Live microphone** — stream transcription from any audio input device with voice activity detection (Silero VAD v5)
- **Daemon mode** — background service controllable via Unix socket or signals, designed for hotkey-triggered dictation (Hammerspoon, skhd, Karabiner)
- **Clipboard output** — optional `--clipboard` flag pastes transcription directly
- **CoreML support** — experimental CoreML acceleration via `--coreml` flag

## Requirements

- **macOS on Apple Silicon** (M1/M2/M3/M4)
- **Rust 1.85+** (edition 2024)
- **~1.3 GB disk space** for FP16 model weights (or ~670 MB for INT8)

> Linux/Intel Mac may work with CPU-only inference but are untested.

## Installation

```bash
git clone https://github.com/lucataco/parakeet-cli.git
cd parakeet-cli
cargo build --release --bin parakeet
```

The binary is at `target/release/parakeet`. Optionally copy it to your PATH:

```bash
cp target/release/parakeet /usr/local/bin/parakeet
```

## Quick Start

```bash
# 1. Download model weights (~1.3 GB FP16)
parakeet download

# 2. Transcribe a file
parakeet transcribe recording.wav

# 3. Live transcription from microphone
parakeet listen
```

## Commands

### `parakeet download`

Download model weights from HuggingFace. Files are cached and skipped if already present.

```
parakeet download [OPTIONS]

Options:
  --model-dir <PATH>   Directory to store model files
                        [default: ~/Library/Application Support/parakeet/models/parakeet-tdt-0.6b-v3]
  --fp16               Download FP16 quantized model (1.2 GB) instead of the default INT8 weights.
  -v, --verbose        Enable verbose output
```

### `parakeet transcribe`

Transcribe a WAV audio file (any sample rate — automatically resampled to 16kHz).

```
parakeet transcribe <FILE> [OPTIONS]

Options:
  --model-dir <PATH>   Directory containing model files
  --format <FORMAT>    Output format: text, json [default: text]
  --coreml             Enable CoreML acceleration (experimental)
  -v, --verbose        Enable verbose output (model details, tensor shapes, timing stats)
```

Example with JSON output:

```bash
parakeet transcribe meeting.wav --format json
```

### `parakeet listen`

Stream transcription from the microphone in real time. Uses Silero VAD v5 to detect speech boundaries — when you stop talking, the utterance is transcribed and printed.

Press `Ctrl-C` to stop.

```
parakeet listen [OPTIONS]

Options:
  --device <NAME>        Audio input device (substring match; use 'devices' to list)
  --model-dir <PATH>     Directory containing model files
  --vad-threshold <F32>  VAD speech probability threshold, 0.0-1.0 [default: 0.5]
  --silence-ms <MS>      Silence duration in ms to end an utterance [default: 1500]
  --clipboard            Copy each transcription to clipboard (via pbcopy)
  --debug                Print debug info: audio levels, VAD probabilities, state transitions
  --coreml               Enable CoreML acceleration (experimental)
  -v, --verbose          Enable verbose output (model details, tensor shapes, timing stats)
```

Example with a specific microphone and tighter VAD:

```bash
parakeet listen --device "MacBook Pro" --vad-threshold 0.6 --silence-ms 1000
```

### `parakeet serve`

Run as a background daemon controllable via Unix socket or Unix signals. Designed for hotkey-triggered dictation workflows.

```
parakeet serve [OPTIONS]

Options:
  --socket <PATH>      Unix socket path [default: ~/Library/Application Support/parakeet/run/daemon.sock]
  --pid-file <PATH>    PID file path [default: ~/Library/Application Support/parakeet/run/daemon.pid]
  --device <NAME>      Audio input device
  --model-dir <PATH>   Directory containing model files
  --clipboard          Copy transcription to clipboard (via pbcopy)
  --coreml             Enable CoreML acceleration (experimental)
  -v, --verbose        Enable verbose output (model details, tensor shapes, timing stats)
```

See [Daemon Mode](#daemon-mode) below for control commands and integration examples.

### `parakeet devices`

List available audio input devices.

```bash
parakeet devices
```

```
Audio input devices:

  MacBook Pro Microphone (default): 1ch, 48000Hz, F32
  External USB Mic: 2ch, 44100Hz, I16
```

## Model Variants

Two quantization levels are available, both based on Parakeet TDT 0.6B v3:

| Variant | Encoder | Decoder | Total | Speed | Download |
|---------|---------|---------|-------|-------|----------|
| **INT8** (default) | 652 MB | 18 MB | ~670 MB | ~50x realtime | `parakeet download` |
| **FP16** | 1.2 GB | 35 MB | ~1.3 GB | ~59x realtime | `parakeet download --fp16` |

The model loader auto-detects which variant is present and prefers FP16 > INT8 > FP32 (legacy). FP32 legacy models may use external data files (`.onnx.data`), which are automatically preloaded.

## Daemon Mode

The `serve` command starts a background process that loads the model once and waits for commands. This avoids the ~1 second model load time on each invocation, making it ideal for hotkey-triggered dictation.

### Socket Commands

By default the daemon uses a private per-user runtime directory. For shell examples below:

```bash
PARAKEET_SOCKET="$HOME/Library/Application Support/parakeet/run/daemon.sock"
PARAKEET_PID="$HOME/Library/Application Support/parakeet/run/daemon.pid"
```

Send commands to `$PARAKEET_SOCKET` as plain text or JSON:

```bash
# Toggle recording on/off
echo "toggle" | nc -U "$PARAKEET_SOCKET"

# Start recording
echo "start" | nc -U "$PARAKEET_SOCKET"

# Stop recording and transcribe
echo "stop" | nc -U "$PARAKEET_SOCKET"

# Check daemon status
echo "status" | nc -U "$PARAKEET_SOCKET"

# Shut down the daemon
echo "shutdown" | nc -U "$PARAKEET_SOCKET"
```

JSON format is also accepted:

```bash
echo '{"command":"toggle"}' | nc -U "$PARAKEET_SOCKET"
```

### Signal Control

| Signal    | Action                      |
|-----------|-----------------------------|
| `SIGUSR1` | Toggle recording on/off    |
| `SIGUSR2` | Stop recording & transcribe |
| `SIGINT`  | Shut down daemon           |

```bash
# Toggle via signal (using PID file)
kill -USR1 $(cat "$PARAKEET_PID")

# Stop recording
kill -USR2 $(cat "$PARAKEET_PID")
```

### Integration Examples

#### Hammerspoon (macOS)

Bind a hotkey to toggle dictation:

```lua
-- ~/.hammerspoon/init.lua
hs.hotkey.bind({"cmd", "shift"}, "D", function()
    hs.execute("echo toggle | nc -U \"$HOME/Library/Application Support/parakeet/run/daemon.sock\"", true)
end)
```

#### skhd

```
# ~/.skhdrc
cmd + shift - d : echo "toggle" | nc -U "$HOME/Library/Application Support/parakeet/run/daemon.sock"
```

#### Karabiner-Elements

Use a complex modification to map a key to:

```bash
echo "toggle" | nc -U "$HOME/Library/Application Support/parakeet/run/daemon.sock"
```

## Architecture

```
src/
├── main.rs              # Entry point, command dispatch
├── cli.rs               # Clap CLI definitions
├── download.rs          # HuggingFace model download (multi-repo)
├── listen.rs            # Live mic → VAD → transcribe pipeline
├── serve.rs             # Daemon mode (socket + signal control)
├── audio/
│   ├── mel.rs           # 128-bin log-mel spectrogram (FFT, Hann window, mel filterbank)
│   ├── resample.rs      # WAV loading, stereo→mono, linear interpolation resampling
│   ├── capture.rs       # Mic capture via cpal (multi-format, multi-channel)
│   └── buffer.rs        # Growable audio buffer for utterance accumulation
├── model/
│   ├── encoder.rs       # ONNX encoder session (CoreML + CPU, external data preloading)
│   ├── decoder.rs       # TDT greedy decoder with LSTM state management
│   └── tokenizer.rs     # SentencePiece vocab, token ID → text
└── vad/
    └── silero.rs        # Silero VAD v5 ONNX, VadSegmenter state machine
```

### Inference Pipeline

**File transcription:**

```
WAV file → resample to 16kHz mono → 128-bin log-mel spectrogram
  → Encoder (ONNX) → TDT Decoder (greedy, autoregressive) → Tokenizer → text
```

**Live streaming:**

```
Microphone → mono f32 chunks → resample to 16kHz
  → Silero VAD (32ms chunks) → detect speech/silence boundaries
  → accumulate utterance → transcribe on speech end → output
```

### Model Details

| Component | File (FP16) | Size |
|-----------|-------------|------|
| Encoder | `encoder-model.fp16.onnx` | 1.2 GB |
| Decoder | `decoder_joint-model.fp16.onnx` | 35 MB |
| Tokenizer | `vocab.txt` (8193 tokens, 25 languages) | 92 KB |
| VAD | `silero_vad.onnx` (auto-downloaded) | 2.3 MB |

- **Encoder input:** 128-bin mel features, 16kHz audio, 25ms window, 10ms hop
- **Decoder:** TDT (Token-and-Duration Transducer) with greedy search and duration-based time stepping
- **Tokenizer:** SentencePiece unigram, 8193 tokens (v3 multilingual) including blank

## Testing & Benchmarks

### Unit Tests

Run the unit tests (no model files required):

```bash
cargo test
```

### Performance Benchmarks

The project includes a comprehensive [Criterion.rs](https://github.com/bheisler/criterion.rs) benchmark suite covering every stage of the STT pipeline.

**Run all benchmarks:**

```bash
cargo bench
```

**Run a specific benchmark suite:**

```bash
cargo bench -- mel_spectrogram     # Mel spectrogram computation (CPU-only)
cargo bench -- resample            # Audio resampling & stereo-to-mono (CPU-only)
cargo bench -- vad                 # Silero VAD inference & segmenter
cargo bench -- encoder             # ONNX encoder inference
cargo bench -- decoder             # TDT greedy decoder loop
cargo bench -- end_to_end          # Full audio-to-text pipeline
cargo bench -- memory              # Memory allocation tracking
```

> Benchmarks that require model files (`encoder`, `decoder`, `end_to_end`, `memory`) will skip gracefully with a message if the model hasn't been downloaded. Run `parakeet download` first to enable them. VAD benchmarks require `silero_vad.onnx`, which is auto-downloaded on the first `parakeet listen` run.

**What's measured:**

| Benchmark Suite | What It Measures | Model Required |
|---|---|---|
| `mel_spectrogram` | FFT + filterbank throughput across 1s-60s audio | No |
| `resampling` | One-shot vs streaming resampling, stereo-to-mono | No |
| `vad` | Per-chunk VAD latency, streaming throughput, segmenter state machine | VAD model |
| `encoder` | Encoder inference time and realtime factor (1s-30s) | Yes |
| `decoder` | Decoder step latency and total decode time | Yes |
| `end_to_end` | Full pipeline RTF, batch throughput (10x5s sequential) | Yes |
| `memory` | Allocation sizes per stage (mel, buffer, encoder, full pipeline) | Partial |

HTML reports are generated in `target/criterion/` for regression tracking across runs.

## Performance

Benchmarked on Apple Silicon (M-series), CPU inference with FP16 model:

| Metric | Value |
|--------|-------|
| Inference speed | ~59x realtime |
| Model load time | ~1.1 seconds |
| Download size | ~1.3 GB (FP16) |
| Streaming latency | Utterance end + ~0.5s transcription |

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

The NVIDIA Parakeet TDT model weights are subject to NVIDIA's [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license. The Silero VAD model is licensed under MIT.
