# parakeet-cli

Local speech-to-text CLI powered by NVIDIA's Parakeet TDT 0.6B v2 model. Runs entirely on-device via ONNX Runtime — no cloud APIs, no network required after model download.

## Features

- **High accuracy** — NVIDIA Parakeet TDT 0.6B v2 (6.05% WER on LibriSpeech test-clean) with punctuation, capitalization, and timestamps
- **Fully local** — all inference runs on-device; audio never leaves your machine
- **Fast** — ~50x realtime on Apple Silicon CPU
- **File transcription** — transcribe WAV files with text or JSON output
- **Live microphone** — stream transcription from any audio input device with voice activity detection (Silero VAD v5)
- **Daemon mode** — background service controllable via Unix socket or signals, designed for hotkey-triggered dictation (Hammerspoon, skhd, Karabiner)
- **Clipboard output** — optional `--clipboard` flag pastes transcription directly

## Requirements

- **macOS on Apple Silicon** (M1/M2/M3/M4) — CPU inference is excellent; CoreML EP is attempted but falls back to CPU gracefully
- **Rust 1.80+** (edition 2024)
- **~2.4 GB disk space** for model weights

> Linux/Intel Mac may work with CPU-only inference but are untested.

## Installation

```bash
git clone https://github.com/lucataco/parakeet-cli.git
cd parakeet-cli
cargo build --release
```

The binary is at `target/release/parakeet-cli`. Optionally copy it to your PATH:

```bash
cp target/release/parakeet-cli /usr/local/bin/parakeet
```

## Quick Start

```bash
# 1. Download model weights (~2.4 GB)
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
                        [default: ~/Library/Application Support/parakeet/models/parakeet-tdt-0.6b-v2]
  --int8               Download INT8 quantized model (smaller, slightly less accurate)
```

### `parakeet transcribe`

Transcribe an audio file (WAV format, any sample rate — automatically resampled to 16kHz).

```
parakeet transcribe <FILE> [OPTIONS]

Options:
  --model-dir <PATH>   Directory containing model files
  --format <FORMAT>    Output format: text, json [default: text]
  --timestamps         Include word-level timestamps (reserved for future use)
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
  --socket <PATH>      Unix socket path [default: /tmp/parakeet.sock]
  --pid-file <PATH>    PID file path [default: /tmp/parakeet.pid]
  --device <NAME>      Audio input device
  --model-dir <PATH>   Directory containing model files
  --clipboard          Copy transcription to clipboard
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

## Daemon Mode

The `serve` command starts a background process that loads the model once and waits for commands. This avoids the ~1 second model load time on each invocation, making it ideal for hotkey-triggered dictation.

### Socket Commands

Send commands to `/tmp/parakeet.sock` as plain text or JSON:

```bash
# Toggle recording on/off
echo "toggle" | nc -U /tmp/parakeet.sock

# Start recording
echo "start" | nc -U /tmp/parakeet.sock

# Stop recording and transcribe
echo "stop" | nc -U /tmp/parakeet.sock

# Check daemon status
echo "status" | nc -U /tmp/parakeet.sock

# Shut down the daemon
echo "shutdown" | nc -U /tmp/parakeet.sock
```

JSON format is also accepted:

```bash
echo '{"command":"toggle"}' | nc -U /tmp/parakeet.sock
```

### Signal Control

| Signal    | Action                      |
|-----------|-----------------------------|
| `SIGUSR1` | Toggle recording on/off    |
| `SIGUSR2` | Stop recording & transcribe |
| `SIGINT`  | Shut down daemon           |

```bash
# Toggle via signal (using PID file)
kill -USR1 $(cat /tmp/parakeet.pid)

# Stop recording
kill -USR2 $(cat /tmp/parakeet.pid)
```

### Integration Examples

#### Hammerspoon (macOS)

Bind a hotkey to toggle dictation:

```lua
-- ~/.hammerspoon/init.lua
hs.hotkey.bind({"cmd", "shift"}, "D", function()
    hs.execute("echo toggle | nc -U /tmp/parakeet.sock", true)
end)
```

#### skhd

```
# ~/.skhdrc
cmd + shift - d : echo "toggle" | nc -U /tmp/parakeet.sock
```

#### Karabiner-Elements

Use a complex modification to map a key to:

```bash
echo "toggle" | nc -U /tmp/parakeet.sock
```

## Architecture

```
src/
├── main.rs              # Entry point, command dispatch
├── cli.rs               # Clap CLI definitions
├── download.rs          # HuggingFace model download
├── listen.rs            # Live mic → VAD → transcribe pipeline
├── serve.rs             # Daemon mode (socket + signal control)
├── audio/
│   ├── mel.rs           # 128-bin log-mel spectrogram (FFT, Hann window, mel filterbank)
│   ├── resample.rs      # WAV loading, stereo→mono, linear interpolation resampling
│   ├── capture.rs       # Mic capture via cpal (multi-format, multi-channel)
│   └── buffer.rs        # Growable audio buffer for utterance accumulation
├── model/
│   ├── encoder.rs       # ONNX encoder session (CoreML → CPU fallback)
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

| Component | File | Size |
|-----------|------|------|
| Encoder | `encoder-model.onnx` + `.data` | ~2.3 GB |
| Decoder | `decoder_joint-model.onnx` | ~34 MB |
| Tokenizer | `vocab.txt` (1025 tokens) | ~9 KB |
| VAD | `silero_vad.onnx` (auto-downloaded) | ~2.3 MB |

- **Encoder input:** 128-bin mel features, 16kHz audio, 25ms window, 10ms hop
- **Decoder:** TDT (Token-and-Duration Transducer) with greedy search and duration-based time stepping
- **Tokenizer:** SentencePiece unigram, 1025 tokens including blank

## Testing & Benchmarks

### Unit Tests

Run the 22 unit tests (no model files required):

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

Benchmarked on Apple Silicon (M-series), CPU inference:

| Metric | Value |
|--------|-------|
| Inference speed | ~50x realtime |
| Model load time | ~1.1 seconds |
| Streaming latency | Utterance end + ~0.5s transcription |

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

The NVIDIA Parakeet TDT model weights are subject to NVIDIA's license terms. The Silero VAD model is licensed under MIT.
