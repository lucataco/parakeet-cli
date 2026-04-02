/// Daemon mode for hotkey-triggered dictation.
///
/// The daemon pre-loads all models, then waits for commands via:
/// - Unix socket in the per-user runtime directory (JSON protocol)
/// - Unix signals: SIGUSR1=toggle recording, SIGUSR2=stop recording
///
/// When recording is triggered, the daemon captures audio from the mic,
/// runs VAD-segmented transcription, and outputs the result to stdout
/// and optionally to the clipboard.
///
/// Designed for integration with Hammerspoon, Karabiner, or skhd:
///   skhd: `ctrl - r : echo '{"command":"toggle"}' | nc -U "$HOME/Library/Application Support/parakeet/run/daemon.sock"`
///   signal: `kill -USR1 $(cat "$HOME/Library/Application Support/parakeet/run/daemon.pid")`
use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
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
const STATE_CAPTURE: u8 = 4;
const STATE_CANCELLING: u8 = 5;

fn state_name(s: u8) -> &'static str {
    match s {
        STATE_IDLE => "idle",
        STATE_RECORDING => "recording",
        STATE_STOPPING => "stopping",
        STATE_SHUTDOWN => "shutdown",
        STATE_CAPTURE => "capturing",
        STATE_CANCELLING => "cancelling",
        _ => "unknown",
    }
}

/// Push samples into a preroll ring buffer, keeping the last `max_samples`.
fn push_preroll(preroll: &mut VecDeque<f32>, chunk: &[f32], max_samples: usize) {
    preroll.extend(chunk.iter().copied());
    while preroll.len() > max_samples {
        preroll.pop_front();
    }
}

/// Shared channel for returning capture results to a waiting socket connection.
type CaptureChannel = Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Sender<CaptureResult>>>>;

/// Result from a single-utterance capture session.
#[derive(Clone, Debug)]
struct CaptureResult {
    text: String,
    duration: f64,
    inference_time: f64,
}

// ── PID file RAII guard ─────────────────────────────────────────────

struct PidFile {
    path: PathBuf,
}

impl PidFile {
    fn create(path: &Path) -> Result<Self> {
        ensure_runtime_parent(path)?;

        let pid = std::process::id();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| {
                format!(
                    "Failed to create PID file {}. Remove it if another daemon is not running.",
                    path.display()
                )
            })?;
        file.write_all(pid.to_string().as_bytes())
            .with_context(|| format!("Failed to write PID file: {}", path.display()))?;
        file.sync_all()
            .with_context(|| format!("Failed to sync PID file: {}", path.display()))?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).with_context(
            || format!("Failed to secure PID file permissions: {}", path.display()),
        )?;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inference_time: Option<f64>,
}

/// Handle a fire-and-forget socket command (toggle, start, stop, status, shutdown, cancel).
async fn handle_quick_command(
    stream: &mut tokio::net::UnixStream,
    command: &str,
    state: &Arc<AtomicU8>,
    _capture_tx: &CaptureChannel,
) -> Result<()> {
    let current = state.load(Ordering::SeqCst);

    let resp = match command {
        "toggle" => match current {
            STATE_IDLE => {
                state.store(STATE_RECORDING, Ordering::SeqCst);
                eprintln!("[daemon] Recording started (socket toggle)");
                SocketResponse {
                    status: "ok".into(),
                    state: Some("recording".into()),
                    message: Some("Recording started".into()),
                    text: None, duration: None, inference_time: None,
                }
            }
            STATE_RECORDING | STATE_CAPTURE => {
                state.store(STATE_STOPPING, Ordering::SeqCst);
                eprintln!("[daemon] Recording stopping (socket toggle)");
                SocketResponse {
                    status: "ok".into(),
                    state: Some("stopping".into()),
                    message: Some("Recording stopping".into()),
                    text: None, duration: None, inference_time: None,
                }
            }
            _ => SocketResponse {
                status: "ok".into(),
                state: Some(state_name(current).into()),
                message: Some("Cannot toggle in current state".into()),
                text: None, duration: None, inference_time: None,
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
                    text: None, duration: None, inference_time: None,
                }
            } else {
                SocketResponse {
                    status: "ok".into(),
                    state: Some(state_name(current).into()),
                    message: Some("Already recording or busy".into()),
                    text: None, duration: None, inference_time: None,
                }
            }
        }
        "stop" => {
            if current == STATE_RECORDING || current == STATE_CAPTURE {
                state.store(STATE_STOPPING, Ordering::SeqCst);
                eprintln!("[daemon] Recording stopping (socket stop)");
                SocketResponse {
                    status: "ok".into(),
                    state: Some("stopping".into()),
                    message: Some("Recording stopping".into()),
                    text: None, duration: None, inference_time: None,
                }
            } else {
                SocketResponse {
                    status: "ok".into(),
                    state: Some(state_name(current).into()),
                    message: Some("Not recording".into()),
                    text: None, duration: None, inference_time: None,
                }
            }
        }
        "cancel" => {
            if current == STATE_RECORDING || current == STATE_CAPTURE {
                state.store(STATE_CANCELLING, Ordering::SeqCst);
                eprintln!("[daemon] Recording cancelling (socket cancel)");
                SocketResponse {
                    status: "ok".into(),
                    state: Some("idle".into()),
                    message: Some("Recording cancelled".into()),
                    text: None, duration: None, inference_time: None,
                }
            } else {
                SocketResponse {
                    status: "ok".into(),
                    state: Some(state_name(current).into()),
                    message: Some("Not recording".into()),
                    text: None, duration: None, inference_time: None,
                }
            }
        }
        "status" => SocketResponse {
            status: "ok".into(),
            state: Some(state_name(current).into()),
            message: None,
            text: None, duration: None, inference_time: None,
        },
        "shutdown" => {
            state.store(STATE_SHUTDOWN, Ordering::SeqCst);
            eprintln!("[daemon] Shutdown requested (socket)");
            SocketResponse {
                status: "ok".into(),
                state: Some("shutdown".into()),
                message: Some("Shutting down".into()),
                text: None, duration: None, inference_time: None,
            }
        }
        _ => SocketResponse {
            status: "error".into(),
            state: None,
            message: Some(format!(
                "Unknown command: {command}. Valid: toggle, start, stop, capture, cancel, status, shutdown"
            )),
            text: None, duration: None, inference_time: None,
        },
    };

    let json = serde_json::to_string(&resp)?;
    stream.write_all(json.as_bytes()).await?;

    Ok(())
}

/// Handle the `capture` command: start recording, wait for one utterance,
/// and return the transcription on the same socket connection.
async fn handle_capture_command(
    stream: &mut tokio::net::UnixStream,
    state: &Arc<AtomicU8>,
    capture_tx: &CaptureChannel,
) -> Result<()> {
    let current = state.load(Ordering::SeqCst);
    if current != STATE_IDLE {
        let resp = SocketResponse {
            status: "error".into(),
            state: Some(state_name(current).into()),
            message: Some("Busy — cannot capture right now".into()),
            text: None, duration: None, inference_time: None,
        };
        let json = serde_json::to_string(&resp)?;
        stream.write_all(json.as_bytes()).await?;
        return Ok(());
    }

    // Set up the oneshot channel for the result
    let (tx, rx) = tokio::sync::oneshot::channel::<CaptureResult>();
    {
        let mut lock = capture_tx.lock().await;
        *lock = Some(tx);
    }

    // Transition to CAPTURE state — main loop will start recording
    state.store(STATE_CAPTURE, Ordering::SeqCst);
    eprintln!("[daemon] Capture started (socket capture)");

    // Wait for the result with a 30-second timeout
    let resp = match tokio::time::timeout(std::time::Duration::from_secs(30), rx).await {
        Ok(Ok(result)) => {
            if result.text.is_empty() {
                SocketResponse {
                    status: "ok".into(),
                    state: Some("idle".into()),
                    message: Some("No speech detected or cancelled".into()),
                    text: None, duration: None, inference_time: None,
                }
            } else {
                SocketResponse {
                    status: "ok".into(),
                    state: Some("idle".into()),
                    message: None,
                    text: Some(result.text),
                    duration: Some(result.duration),
                    inference_time: Some(result.inference_time),
                }
            }
        }
        Ok(Err(_)) => {
            // Sender was dropped (e.g., cancel or shutdown)
            SocketResponse {
                status: "error".into(),
                state: Some(state_name(state.load(Ordering::SeqCst)).into()),
                message: Some("Capture aborted".into()),
                text: None, duration: None, inference_time: None,
            }
        }
        Err(_) => {
            // Timeout — cancel the capture
            state.store(STATE_CANCELLING, Ordering::SeqCst);
            SocketResponse {
                status: "error".into(),
                state: Some("idle".into()),
                message: Some("Capture timed out after 30s".into()),
                text: None, duration: None, inference_time: None,
            }
        }
    };

    let json = serde_json::to_string(&resp)?;
    stream.write_all(json.as_bytes()).await?;

    Ok(())
}

/// Handle a single socket connection: parse the command and dispatch.
async fn handle_socket_connection(
    stream: &mut tokio::net::UnixStream,
    state: &Arc<AtomicU8>,
    capture_tx: &CaptureChannel,
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
                    text: None, duration: None, inference_time: None,
                };
                let json = serde_json::to_string(&resp)?;
                stream.write_all(json.as_bytes()).await?;
                return Ok(());
            }
        }
    } else {
        input.to_string()
    };

    // `capture` holds the connection open, everything else is fire-and-forget
    if command == "capture" {
        handle_capture_command(stream, state, capture_tx).await
    } else {
        handle_quick_command(stream, &command, state, capture_tx).await
    }
}

fn ensure_runtime_parent(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("Path has no parent directory: {}", path.display()))?;

    if !parent.exists() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create runtime directory: {}", parent.display()))?;
        std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700)).with_context(
            || {
                format!(
                    "Failed to secure runtime directory permissions: {}",
                    parent.display()
                )
            },
        )?;
    }

    Ok(())
}

fn prepare_socket_path(socket_path: &Path) -> Result<()> {
    ensure_runtime_parent(socket_path)?;

    if !socket_path.exists() {
        return Ok(());
    }

    let metadata = std::fs::symlink_metadata(socket_path)
        .with_context(|| format!("Failed to inspect socket path: {}", socket_path.display()))?;

    if metadata.file_type().is_socket() {
        std::fs::remove_file(socket_path)
            .with_context(|| format!("Failed to remove stale socket: {}", socket_path.display()))?;
        return Ok(());
    }

    anyhow::bail!(
        "Refusing to replace non-socket file at {}. Choose a different --socket path or remove the file manually.",
        socket_path.display(),
    );
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
    use_coreml: bool,
) -> Result<()> {
    // Write PID file (cleaned up on drop)
    let _pid_guard = PidFile::create(pid_file_path)?;

    // Download VAD model if needed
    let vad_path = vad::ensure_vad_model(model_dir).await?;

    // Load Parakeet model
    eprintln!("[daemon] Loading Parakeet model...");
    let mut model = ParakeetModel::load(model_dir, use_coreml, verbose)?;
    eprintln!();

    // Load Silero VAD
    let mut vad_model = SileroVad::load(&vad_path, verbose)?;
    eprintln!();

    // Shared state
    let state = Arc::new(AtomicU8::new(STATE_IDLE));
    let capture_tx: CaptureChannel = Arc::new(tokio::sync::Mutex::new(None));

    // ── Start Unix socket listener ──────────────────────────────────
    prepare_socket_path(socket_path)?;

    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("Failed to bind Unix socket: {}", socket_path.display()))?;
    std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600)).with_context(
        || {
            format!(
                "Failed to secure socket permissions: {}",
                socket_path.display()
            )
        },
    )?;

    let _socket_guard = SocketGuard {
        path: socket_path.to_path_buf(),
    };

    eprintln!("[daemon] Listening on socket: {}", socket_path.display());

    // Spawn socket accept loop — each connection is spawned as its own task
    // so that `capture` (which holds the connection open) doesn't block
    // other commands like `cancel`.
    let socket_state = state.clone();
    let socket_capture_tx = capture_tx.clone();
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((mut stream, _)) => {
                    let st = socket_state.clone();
                    let ct = socket_capture_tx.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_socket_connection(&mut stream, &st, &ct).await {
                            eprintln!("[daemon] Socket error: {e}");
                        }
                    });
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

        let mut sigusr1 = match signal(SignalKind::user_defined1()) {
            Ok(signal) => signal,
            Err(e) => {
                eprintln!("[daemon] Failed to register SIGUSR1: {e}");
                return;
            }
        };
        let mut sigusr2 = match signal(SignalKind::user_defined2()) {
            Ok(signal) => signal,
            Err(e) => {
                eprintln!("[daemon] Failed to register SIGUSR2: {e}");
                return;
            }
        };

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

        if current != STATE_RECORDING && current != STATE_CAPTURE {
            // Idle or other non-recording state — sleep briefly and poll again
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            continue;
        }

        let is_capture_mode = current == STATE_CAPTURE;

        // ── Recording session ───────────────────────────────────────
        eprintln!(
            "[daemon] Starting {} session...",
            if is_capture_mode { "capture" } else { "recording" }
        );

        // Start audio capture for this session
        let capture = match audio::start_capture(device) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[daemon] Failed to start capture: {e}");
                // If capture mode, send error through channel
                if is_capture_mode {
                    if let Some(tx) = capture_tx.lock().await.take() {
                        let _ = tx.send(CaptureResult {
                            text: String::new(),
                            duration: 0.0,
                            inference_time: 0.0,
                        });
                    }
                }
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

        // Preroll buffer: keeps the last 200ms of audio so speech onset isn't clipped
        let mut preroll: VecDeque<f32> = VecDeque::new();
        let preroll_samples = (0.2 * 16000.0) as usize; // 3200 samples

        // Accumulate all transcriptions from this session
        let mut session_text = String::new();
        let mut session_duration: f64 = 0.0;
        let mut session_infer_time: f64 = 0.0;
        let mut got_first_utterance = false;

        // Recording loop — runs until state changes
        loop {
            let current = state.load(Ordering::SeqCst);
            if current == STATE_STOPPING
                || current == STATE_CANCELLING
                || current == STATE_SHUTDOWN
            {
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

                // Always maintain the preroll buffer
                push_preroll(&mut preroll, &vad_chunk, preroll_samples);

                let event = segmenter.process(speech_prob);

                match event {
                    VadEvent::SpeechStart => {
                        utterance_buffer.clear();
                        // Use preroll to capture audio before VAD triggered
                        let preroll_vec: Vec<f32> = preroll.iter().copied().collect();
                        utterance_buffer.push(&preroll_vec);
                        preroll.clear();
                    }
                    VadEvent::SpeechEnd => {
                        let duration_secs = utterance_buffer.duration_secs();
                        let samples = utterance_buffer.drain();
                        if samples.len() > 1600 {
                            let features = audio::compute_mel_spectrogram(&samples, &mel_config);
                            let infer_start = std::time::Instant::now();
                            match model.transcribe(&features) {
                                Ok(text) => {
                                    let infer_time = infer_start.elapsed().as_secs_f64();
                                    let text = text.trim().to_string();
                                    if !text.is_empty() {
                                        eprintln!("[daemon] Transcribed: {}", text);
                                        if !session_text.is_empty() {
                                            session_text.push(' ');
                                        }
                                        session_text.push_str(&text);
                                        session_duration += duration_secs as f64;
                                        session_infer_time += infer_time;
                                        got_first_utterance = true;

                                        // In capture mode, exit after first utterance
                                        if is_capture_mode {
                                            break;
                                        }
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

            // Break outer loop too if capture mode got its utterance
            if is_capture_mode && got_first_utterance {
                break;
            }
        }

        // ── Handle cancellation ─────────────────────────────────────
        let current_after = state.load(Ordering::SeqCst);
        if current_after == STATE_CANCELLING {
            eprintln!("[daemon] Session cancelled, discarding audio.");
            utterance_buffer.clear();

            // If capture mode, send empty result through channel
            if is_capture_mode {
                if let Some(tx) = capture_tx.lock().await.take() {
                    let _ = tx.send(CaptureResult {
                        text: String::new(),
                        duration: 0.0,
                        inference_time: 0.0,
                    });
                }
            }

            drop(capture);
            state.store(STATE_IDLE, Ordering::SeqCst);
            eprintln!("[daemon] Session cancelled. Ready for next command.");
            continue;
        }

        // ── Flush remaining audio (non-cancel path) ─────────────────
        if !is_capture_mode || !got_first_utterance {
            let mut remaining = resampler.finish();
            if !remaining.is_empty() {
                vad_buf.append(&mut remaining);
            }
            while vad_buf.len() >= VAD_CHUNK_SAMPLES {
                let vad_chunk: Vec<f32> = vad_buf.drain(..VAD_CHUNK_SAMPLES).collect();
                let speech_prob = vad_model.process_chunk(&vad_chunk)?;
                push_preroll(&mut preroll, &vad_chunk, preroll_samples);
                let event = segmenter.process(speech_prob);

                match event {
                    VadEvent::SpeechStart => {
                        utterance_buffer.clear();
                        let preroll_vec: Vec<f32> = preroll.iter().copied().collect();
                        utterance_buffer.push(&preroll_vec);
                        preroll.clear();
                    }
                    VadEvent::SpeechEnd => {
                        let duration_secs = utterance_buffer.duration_secs();
                        let samples = utterance_buffer.drain();
                        if samples.len() > 1600 {
                            let features = audio::compute_mel_spectrogram(&samples, &mel_config);
                            let infer_start = std::time::Instant::now();
                            match model.transcribe(&features) {
                                Ok(text) => {
                                    let infer_time = infer_start.elapsed().as_secs_f64();
                                    let text = text.trim().to_string();
                                    if !text.is_empty() {
                                        eprintln!("[daemon] Transcribed: {}", text);
                                        if !session_text.is_empty() {
                                            session_text.push(' ');
                                        }
                                        session_text.push_str(&text);
                                        session_duration += duration_secs as f64;
                                        session_infer_time += infer_time;
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

            // Transcribe any remaining buffered audio
            if utterance_buffer.duration_secs() > 0.1 {
                let duration_secs = utterance_buffer.duration_secs();
                let samples = utterance_buffer.drain();
                if samples.len() > 1600 {
                    let features = audio::compute_mel_spectrogram(&samples, &mel_config);
                    let infer_start = std::time::Instant::now();
                    match model.transcribe(&features) {
                        Ok(text) => {
                            let infer_time = infer_start.elapsed().as_secs_f64();
                            let text = text.trim().to_string();
                            if !text.is_empty() {
                                eprintln!("[daemon] Transcribed (final): {}", text);
                                if !session_text.is_empty() {
                                    session_text.push(' ');
                                }
                                session_text.push_str(&text);
                                session_duration += duration_secs as f64;
                                session_infer_time += infer_time;
                            }
                        }
                        Err(e) => {
                            eprintln!("[daemon] Final transcription error: {e}");
                        }
                    }
                }
            }
        }

        // ── End of recording session ────────────────────────────────

        // If capture mode, send result through the channel
        if is_capture_mode {
            if let Some(tx) = capture_tx.lock().await.take() {
                let _ = tx.send(CaptureResult {
                    text: session_text.clone(),
                    duration: session_duration,
                    inference_time: session_infer_time,
                });
            }
        }

        // Output result to stdout (for non-capture mode, or both)
        if !session_text.is_empty() {
            if !is_capture_mode {
                println!("{}", session_text);
            }

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
