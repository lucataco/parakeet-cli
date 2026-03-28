/// Daemon mode for hotkey-triggered dictation.
///
/// The daemon pre-loads all models, then waits for commands via:
/// - Unix socket at /tmp/parakeet.sock (JSON protocol)
/// - Unix signals: SIGUSR1=toggle recording, SIGUSR2=stop recording
///
/// When recording is triggered, the daemon captures audio from the mic,
/// runs VAD-segmented transcription, and outputs the result to stdout
/// and optionally to the clipboard.
///
/// Designed for integration with Hammerspoon, Karabiner, or skhd:
///   skhd: `ctrl - r : echo '{"command":"toggle"}' | nc -U /tmp/parakeet.sock`
///   signal: `kill -USR1 $(cat /tmp/parakeet.pid)`
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;

use crate::audio::{self, AudioBuffer};
use crate::model::ParakeetModel;
use crate::vad::{self, SileroVad, VAD_CHUNK_SAMPLES, VadEvent, VadSegmenter, VadState};

// ── Daemon state machine ────────────────────────────────────────────

/// Daemon recording state, stored as AtomicU8 for lock-free sharing.
const STATE_IDLE: u8 = 0;
const STATE_RECORDING: u8 = 1;
const STATE_STOPPING: u8 = 2;
const STATE_SHUTDOWN: u8 = 3;

fn state_name(s: u8) -> &'static str {
    match s {
        STATE_IDLE => "idle",
        STATE_RECORDING => "recording",
        STATE_STOPPING => "stopping",
        STATE_SHUTDOWN => "shutdown",
        _ => "unknown",
    }
}

// ── PID file RAII guard ─────────────────────────────────────────────

struct PidFile {
    path: PathBuf,
}

impl PidFile {
    fn create(path: &Path) -> Result<Self> {
        let pid = std::process::id();
        std::fs::write(path, pid.to_string())
            .with_context(|| format!("Failed to write PID file: {}", path.display()))?;
        eprintln!("PID file: {} (pid={})", path.display(), pid);
        Ok(Self {
            path: path.to_path_buf(),
        })
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// ── Socket cleanup RAII guard ───────────────────────────────────────

struct SocketGuard {
    path: PathBuf,
}

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// ── Clipboard output ────────────────────────────────────────────────

fn copy_to_clipboard(text: &str) -> Result<()> {
    use std::process::{Command, Stdio};

    let mut child = Command::new("pbcopy")
        .stdin(Stdio::piped())
        .spawn()
        .context("Failed to spawn pbcopy (macOS only)")?;

    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        stdin.write_all(text.as_bytes())?;
    }

    child.wait()?;
    Ok(())
}

// ── Socket protocol ─────────────────────────────────────────────────

#[derive(serde::Deserialize, Debug)]
struct SocketCommand {
    command: String,
}

#[derive(serde::Serialize)]
struct SocketResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

/// Handle a single socket connection.
async fn handle_socket_connection(
    stream: &mut tokio::net::UnixStream,
    state: &Arc<AtomicU8>,
) -> Result<()> {
    let mut buf = vec![0u8; 1024];
    let n = stream.read(&mut buf).await?;
    if n == 0 {
        return Ok(());
    }

    let input = String::from_utf8_lossy(&buf[..n]);
    let input = input.trim();

    // Support both bare commands ("toggle") and JSON ({"command":"toggle"})
    let command = if input.starts_with('{') {
        match serde_json::from_str::<SocketCommand>(input) {
            Ok(cmd) => cmd.command,
            Err(e) => {
                let resp = SocketResponse {
                    status: "error".into(),
                    state: None,
                    message: Some(format!("Invalid JSON: {e}")),
                };
                let json = serde_json::to_string(&resp)?;
                stream.write_all(json.as_bytes()).await?;
                return Ok(());
            }
        }
    } else {
        input.to_string()
    };

    let current = state.load(Ordering::SeqCst);

    let resp = match command.as_str() {
        "toggle" => match current {
            STATE_IDLE => {
                state.store(STATE_RECORDING, Ordering::SeqCst);
                eprintln!("[daemon] Recording started (socket toggle)");
                SocketResponse {
                    status: "ok".into(),
                    state: Some("recording".into()),
                    message: Some("Recording started".into()),
                }
            }
            STATE_RECORDING => {
                state.store(STATE_STOPPING, Ordering::SeqCst);
                eprintln!("[daemon] Recording stopping (socket toggle)");
                SocketResponse {
                    status: "ok".into(),
                    state: Some("stopping".into()),
                    message: Some("Recording stopping".into()),
                }
            }
            _ => SocketResponse {
                status: "ok".into(),
                state: Some(state_name(current).into()),
                message: Some("Cannot toggle in current state".into()),
            },
        },
        "start" => {
            if current == STATE_IDLE {
                state.store(STATE_RECORDING, Ordering::SeqCst);
                eprintln!("[daemon] Recording started (socket start)");
                SocketResponse {
                    status: "ok".into(),
                    state: Some("recording".into()),
                    message: Some("Recording started".into()),
                }
            } else {
                SocketResponse {
                    status: "ok".into(),
                    state: Some(state_name(current).into()),
                    message: Some("Already recording or busy".into()),
                }
            }
        }
        "stop" => {
            if current == STATE_RECORDING {
                state.store(STATE_STOPPING, Ordering::SeqCst);
                eprintln!("[daemon] Recording stopping (socket stop)");
                SocketResponse {
                    status: "ok".into(),
                    state: Some("stopping".into()),
                    message: Some("Recording stopping".into()),
                }
            } else {
                SocketResponse {
                    status: "ok".into(),
                    state: Some(state_name(current).into()),
                    message: Some("Not recording".into()),
                }
            }
        }
        "status" => SocketResponse {
            status: "ok".into(),
            state: Some(state_name(current).into()),
            message: None,
        },
        "shutdown" => {
            state.store(STATE_SHUTDOWN, Ordering::SeqCst);
            eprintln!("[daemon] Shutdown requested (socket)");
            SocketResponse {
                status: "ok".into(),
                state: Some("shutdown".into()),
                message: Some("Shutting down".into()),
            }
        }
        _ => SocketResponse {
            status: "error".into(),
            state: None,
            message: Some(format!(
                "Unknown command: {command}. Valid: toggle, start, stop, status, shutdown"
            )),
        },
    };

    let json = serde_json::to_string(&resp)?;
    stream.write_all(json.as_bytes()).await?;

    Ok(())
}

// ── Main daemon entry point ─────────────────────────────────────────

/// Run the daemon.
pub async fn run_serve(
    socket_path: &Path,
    pid_file_path: &Path,
    device: &Option<String>,
    model_dir: &Path,
    clipboard: bool,
    verbose: bool,
) -> Result<()> {
    // Write PID file (cleaned up on drop)
    let _pid_guard = PidFile::create(pid_file_path)?;

    // Download VAD model if needed
    let vad_path = vad::ensure_vad_model(model_dir).await?;

    // Load Parakeet model
    eprintln!("[daemon] Loading Parakeet model...");
    let mut model = ParakeetModel::load(model_dir, true, verbose)?;
    eprintln!();

    // Load Silero VAD
    let mut vad_model = SileroVad::load(&vad_path, verbose)?;
    eprintln!();

    // Shared state
    let state = Arc::new(AtomicU8::new(STATE_IDLE));

    // ── Start Unix socket listener ──────────────────────────────────
    // Remove stale socket file
    if socket_path.exists() {
        std::fs::remove_file(socket_path)?;
    }

    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("Failed to bind Unix socket: {}", socket_path.display()))?;

    let _socket_guard = SocketGuard {
        path: socket_path.to_path_buf(),
    };

    eprintln!("[daemon] Listening on socket: {}", socket_path.display());

    // Spawn socket accept loop
    let socket_state = state.clone();
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((mut stream, _)) => {
                    if let Err(e) = handle_socket_connection(&mut stream, &socket_state).await {
                        eprintln!("[daemon] Socket error: {e}");
                    }
                }
                Err(e) => {
                    eprintln!("[daemon] Socket accept error: {e}");
                }
            }
        }
    });

    // ── Start signal handler ────────────────────────────────────────
    let signal_state = state.clone();
    tokio::spawn(async move {
        use tokio::signal::unix::{SignalKind, signal};

        let mut sigusr1 = signal(SignalKind::user_defined1()).expect("Failed to register SIGUSR1");
        let mut sigusr2 = signal(SignalKind::user_defined2()).expect("Failed to register SIGUSR2");

        loop {
            tokio::select! {
                _ = sigusr1.recv() => {
                    let current = signal_state.load(Ordering::SeqCst);
                    match current {
                        STATE_IDLE => {
                            signal_state.store(STATE_RECORDING, Ordering::SeqCst);
                            eprintln!("[daemon] Recording started (SIGUSR1 toggle)");
                        }
                        STATE_RECORDING => {
                            signal_state.store(STATE_STOPPING, Ordering::SeqCst);
                            eprintln!("[daemon] Recording stopping (SIGUSR1 toggle)");
                        }
                        _ => {}
                    }
                }
                _ = sigusr2.recv() => {
                    let current = signal_state.load(Ordering::SeqCst);
                    if current == STATE_RECORDING {
                        signal_state.store(STATE_STOPPING, Ordering::SeqCst);
                        eprintln!("[daemon] Recording stopping (SIGUSR2 stop)");
                    }
                }
            }
        }
    });

    // ── Set up Ctrl-C for clean shutdown ────────────────────────────
    let ctrlc_state = state.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        eprintln!("\n[daemon] Ctrl-C received, shutting down...");
        ctrlc_state.store(STATE_SHUTDOWN, Ordering::SeqCst);
    });

    // ── Main audio processing loop ──────────────────────────────────
    eprintln!("[daemon] Ready. Waiting for commands...");
    eprintln!(
        "[daemon] Send: echo 'toggle' | nc -U {}",
        socket_path.display()
    );
    eprintln!("[daemon] Or:   kill -USR1 {}", std::process::id());
    eprintln!();

    let mel_config = audio::MelConfig::default();

    // The capture stream and processing state are created/destroyed per recording session
    loop {
        let current = state.load(Ordering::SeqCst);

        if current == STATE_SHUTDOWN {
            break;
        }

        if current != STATE_RECORDING {
            // Idle — sleep briefly and poll again
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            continue;
        }

        // ── Recording session ───────────────────────────────────────
        eprintln!("[daemon] Starting recording session...");

        // Start audio capture for this session
        let capture = match audio::start_capture(device) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[daemon] Failed to start capture: {e}");
                state.store(STATE_IDLE, Ordering::SeqCst);
                continue;
            }
        };
        let capture_rate = capture.sample_rate;

        // Reset VAD state for new session
        vad_model.reset();
        let mut segmenter = VadSegmenter::new(0.5, 1500);

        let mut utterance_buffer = AudioBuffer::new(60.0);
        let mut resampler = audio::StreamingResampler::new(capture_rate, 16000);
        let mut vad_buf: Vec<f32> = Vec::new();

        // Accumulate all transcriptions from this session
        let mut session_text = String::new();

        // Recording loop — runs until state changes from RECORDING
        loop {
            let current = state.load(Ordering::SeqCst);
            if current == STATE_STOPPING || current == STATE_SHUTDOWN {
                break;
            }

            // Receive audio
            let chunk = match capture
                .receiver
                .recv_timeout(std::time::Duration::from_millis(50))
            {
                Ok(chunk) => chunk,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    eprintln!("[daemon] Audio stream disconnected");
                    break;
                }
            };

            let resampled = resampler.process(&chunk.samples);

            vad_buf.extend_from_slice(&resampled);

            // Process VAD chunks
            while vad_buf.len() >= VAD_CHUNK_SAMPLES {
                let vad_chunk: Vec<f32> = vad_buf.drain(..VAD_CHUNK_SAMPLES).collect();
                let speech_prob = vad_model.process_chunk(&vad_chunk)?;
                let event = segmenter.process(speech_prob);

                match event {
                    VadEvent::SpeechStart => {
                        utterance_buffer.clear();
                        utterance_buffer.push(&vad_chunk);
                    }
                    VadEvent::SpeechEnd => {
                        let samples = utterance_buffer.drain();
                        if samples.len() > 1600 {
                            let features = audio::compute_mel_spectrogram(&samples, &mel_config);
                            match model.transcribe(&features) {
                                Ok(text) => {
                                    let text = text.trim().to_string();
                                    if !text.is_empty() {
                                        eprintln!("[daemon] Transcribed: {}", text);
                                        if !session_text.is_empty() {
                                            session_text.push(' ');
                                        }
                                        session_text.push_str(&text);
                                    }
                                }
                                Err(e) => {
                                    eprintln!("[daemon] Transcription error: {e}");
                                }
                            }
                        }
                    }
                    VadEvent::None => {
                        if segmenter.state() == VadState::Speaking {
                            utterance_buffer.push(&vad_chunk);
                        }
                    }
                }
            }
        }

        let mut remaining = resampler.finish();
        if !remaining.is_empty() {
            vad_buf.append(&mut remaining);
        }
        while vad_buf.len() >= VAD_CHUNK_SAMPLES {
            let vad_chunk: Vec<f32> = vad_buf.drain(..VAD_CHUNK_SAMPLES).collect();
            let speech_prob = vad_model.process_chunk(&vad_chunk)?;
            let event = segmenter.process(speech_prob);

            match event {
                VadEvent::SpeechStart => {
                    utterance_buffer.clear();
                    utterance_buffer.push(&vad_chunk);
                }
                VadEvent::SpeechEnd => {
                    let samples = utterance_buffer.drain();
                    if samples.len() > 1600 {
                        let features = audio::compute_mel_spectrogram(&samples, &mel_config);
                        match model.transcribe(&features) {
                            Ok(text) => {
                                let text = text.trim().to_string();
                                if !text.is_empty() {
                                    eprintln!("[daemon] Transcribed: {}", text);
                                    if !session_text.is_empty() {
                                        session_text.push(' ');
                                    }
                                    session_text.push_str(&text);
                                }
                            }
                            Err(e) => {
                                eprintln!("[daemon] Transcription error: {e}");
                            }
                        }
                    }
                }
                VadEvent::None => {
                    if segmenter.state() == VadState::Speaking {
                        utterance_buffer.push(&vad_chunk);
                    }
                }
            }
        }

        // ── End of recording session ────────────────────────────────

        // Transcribe any remaining buffered audio
        if utterance_buffer.duration_secs() > 0.1 {
            let samples = utterance_buffer.drain();
            if samples.len() > 1600 {
                let features = audio::compute_mel_spectrogram(&samples, &mel_config);
                match model.transcribe(&features) {
                    Ok(text) => {
                        let text = text.trim().to_string();
                        if !text.is_empty() {
                            eprintln!("[daemon] Transcribed (final): {}", text);
                            if !session_text.is_empty() {
                                session_text.push(' ');
                            }
                            session_text.push_str(&text);
                        }
                    }
                    Err(e) => {
                        eprintln!("[daemon] Final transcription error: {e}");
                    }
                }
            }
        }

        // Output result
        if !session_text.is_empty() {
            println!("{}", session_text);

            if clipboard {
                match copy_to_clipboard(&session_text) {
                    Ok(()) => eprintln!("[daemon] Copied to clipboard"),
                    Err(e) => eprintln!("[daemon] Clipboard error: {e}"),
                }
            }
        } else {
            eprintln!("[daemon] No speech detected in session");
        }

        // Drop the capture stream (stops mic)
        drop(capture);

        // Return to idle (unless shutting down)
        let current = state.load(Ordering::SeqCst);
        if current != STATE_SHUTDOWN {
            state.store(STATE_IDLE, Ordering::SeqCst);
            eprintln!("[daemon] Session ended. Ready for next command.");
        }
    }

    eprintln!("[daemon] Shutting down.");
    Ok(())
}
