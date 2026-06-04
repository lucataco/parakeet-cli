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
use crate::clipboard::copy_text;
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

/// Shared channel for returning capture results to a waiting socket connection.
type CaptureChannel = Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Sender<CaptureResult>>>>;

/// Result from a single-utterance capture session.
#[derive(Clone, Debug)]
struct CaptureResult {
    text: String,
    duration: f64,
    inference_time: f64,
}

impl CaptureResult {
    fn empty() -> Self {
        Self {
            text: String::new(),
            duration: 0.0,
            inference_time: 0.0,
        }
    }
}

#[derive(Debug)]
struct TranscribedUtterance {
    text: String,
    duration: f64,
    inference_time: f64,
}

#[derive(Default)]
struct SessionTranscript {
    text: String,
    duration: f64,
    inference_time: f64,
}

impl SessionTranscript {
    fn append(&mut self, utterance: TranscribedUtterance) {
        if !self.text.is_empty() {
            self.text.push(' ');
        }

        self.text.push_str(&utterance.text);
        self.duration += utterance.duration;
        self.inference_time += utterance.inference_time;
    }

    fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    fn capture_result(&self) -> CaptureResult {
        CaptureResult {
            text: self.text.clone(),
            duration: self.duration,
            inference_time: self.inference_time,
        }
    }
}

fn transcribe_buffer(
    model: &mut ParakeetModel,
    mel_config: &audio::MelConfig,
    utterance_buffer: &mut AudioBuffer,
) -> Result<Option<TranscribedUtterance>> {
    let duration = utterance_buffer.duration_secs() as f64;
    let samples = utterance_buffer.drain();
    transcribe_samples(model, mel_config, &samples, duration)
}

fn transcribe_samples(
    model: &mut ParakeetModel,
    mel_config: &audio::MelConfig,
    samples: &[f32],
    duration: f64,
) -> Result<Option<TranscribedUtterance>> {
    if samples.len() <= audio::MIN_UTTERANCE_SAMPLES {
        return Ok(None);
    }

    let features = audio::compute_mel_spectrogram(samples, mel_config);
    let infer_start = std::time::Instant::now();
    let text = model.transcribe(&features)?.trim().to_string();

    if text.is_empty() {
        return Ok(None);
    }

    Ok(Some(TranscribedUtterance {
        text,
        duration,
        inference_time: infer_start.elapsed().as_secs_f64(),
    }))
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

impl SocketResponse {
    fn new(status: &str) -> Self {
        Self {
            status: status.to_string(),
            state: None,
            message: None,
            text: None,
            duration: None,
            inference_time: None,
        }
    }

    fn ok(state: Option<&str>, message: Option<&str>) -> Self {
        Self {
            state: state.map(str::to_string),
            message: message.map(str::to_string),
            ..Self::new("ok")
        }
    }

    fn error(message: impl Into<String>) -> Self {
        Self {
            message: Some(message.into()),
            ..Self::new("error")
        }
    }

    fn error_with_state(state: &str, message: impl Into<String>) -> Self {
        Self {
            state: Some(state.to_string()),
            message: Some(message.into()),
            ..Self::new("error")
        }
    }

    fn capture_result(result: CaptureResult) -> Self {
        Self {
            state: Some("idle".to_string()),
            text: Some(result.text),
            duration: Some(result.duration),
            inference_time: Some(result.inference_time),
            ..Self::new("ok")
        }
    }
}

/// Handle a fire-and-forget socket command (toggle, start, stop, status, shutdown, cancel).
async fn handle_quick_command(
    stream: &mut tokio::net::UnixStream,
    command: &str,
    state: &Arc<AtomicU8>,
) -> Result<()> {
    let current = state.load(Ordering::SeqCst);

    let resp = match command {
        "toggle" => match current {
            STATE_IDLE => {
                state.store(STATE_RECORDING, Ordering::SeqCst);
                eprintln!("[daemon] Recording started (socket toggle)");
                SocketResponse::ok(Some("recording"), Some("Recording started"))
            }
            STATE_RECORDING | STATE_CAPTURE => {
                state.store(STATE_STOPPING, Ordering::SeqCst);
                eprintln!("[daemon] Recording stopping (socket toggle)");
                SocketResponse::ok(Some("stopping"), Some("Recording stopping"))
            }
            _ => SocketResponse::ok(
                Some(state_name(current)),
                Some("Cannot toggle in current state"),
            ),
        },
        "start" => {
            if current == STATE_IDLE {
                state.store(STATE_RECORDING, Ordering::SeqCst);
                eprintln!("[daemon] Recording started (socket start)");
                SocketResponse::ok(Some("recording"), Some("Recording started"))
            } else {
                SocketResponse::ok(Some(state_name(current)), Some("Already recording or busy"))
            }
        }
        "stop" => {
            if current == STATE_RECORDING || current == STATE_CAPTURE {
                state.store(STATE_STOPPING, Ordering::SeqCst);
                eprintln!("[daemon] Recording stopping (socket stop)");
                SocketResponse::ok(Some("stopping"), Some("Recording stopping"))
            } else {
                SocketResponse::ok(Some(state_name(current)), Some("Not recording"))
            }
        }
        "cancel" => {
            if current == STATE_RECORDING || current == STATE_CAPTURE {
                state.store(STATE_CANCELLING, Ordering::SeqCst);
                eprintln!("[daemon] Recording cancelling (socket cancel)");
                SocketResponse::ok(Some("idle"), Some("Recording cancelled"))
            } else {
                SocketResponse::ok(Some(state_name(current)), Some("Not recording"))
            }
        }
        "status" => SocketResponse::ok(Some(state_name(current)), None),
        "shutdown" => {
            state.store(STATE_SHUTDOWN, Ordering::SeqCst);
            eprintln!("[daemon] Shutdown requested (socket)");
            SocketResponse::ok(Some("shutdown"), Some("Shutting down"))
        }
        _ => SocketResponse::error(format!(
            "Unknown command: {command}. Valid: toggle, start, stop, capture, cancel, status, shutdown"
        )),
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
        let resp = SocketResponse::error_with_state(
            state_name(current),
            "Busy — cannot capture right now",
        );
        let json = serde_json::to_string(&resp)?;
        stream.write_all(json.as_bytes()).await?;
        return Ok(());
    }

    // Set up the oneshot channel for the result
    let (tx, rx) = tokio::sync::oneshot::channel::<CaptureResult>();
    let claim_error = {
        let mut lock = capture_tx.lock().await;

        if lock.is_some() {
            Some(SocketResponse::error_with_state(
                "capturing",
                "Busy — capture already pending",
            ))
        } else if let Err(current) = state.compare_exchange(
            STATE_IDLE,
            STATE_CAPTURE,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Some(SocketResponse::error_with_state(
                state_name(current),
                "Busy — cannot capture right now",
            ))
        } else {
            *lock = Some(tx);
            None
        }
    };

    if let Some(resp) = claim_error {
        let json = serde_json::to_string(&resp)?;
        stream.write_all(json.as_bytes()).await?;
        return Ok(());
    }

    eprintln!("[daemon] Capture started (socket capture)");

    // Wait for the result with a 30-second timeout
    let resp = match tokio::time::timeout(std::time::Duration::from_secs(30), rx).await {
        Ok(Ok(result)) => {
            if result.text.is_empty() {
                SocketResponse::ok(Some("idle"), Some("No speech detected or cancelled"))
            } else {
                SocketResponse::capture_result(result)
            }
        }
        Ok(Err(_)) => {
            // Sender was dropped (e.g., cancel or shutdown)
            SocketResponse::error_with_state(
                state_name(state.load(Ordering::SeqCst)),
                "Capture aborted",
            )
        }
        Err(_) => {
            // Timeout — cancel the capture
            state.store(STATE_CANCELLING, Ordering::SeqCst);
            SocketResponse::error_with_state("idle", "Capture timed out after 30s")
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
                let resp = SocketResponse::error(format!("Invalid JSON: {e}"));
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
        handle_quick_command(stream, &command, state).await
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

        if current == STATE_STOPPING || current == STATE_CANCELLING {
            if let Some(tx) = capture_tx.lock().await.take() {
                let _ = tx.send(CaptureResult::empty());
            }
            state.store(STATE_IDLE, Ordering::SeqCst);
            eprintln!("[daemon] No active session to stop or cancel. Ready for next command.");
            continue;
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
            if is_capture_mode {
                "capture"
            } else {
                "recording"
            }
        );

        // Start audio capture for this session
        let capture = match audio::start_capture(device) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[daemon] Failed to start capture: {e}");
                // If capture mode, send error through channel
                if is_capture_mode {
                    if let Some(tx) = capture_tx.lock().await.take() {
                        let _ = tx.send(CaptureResult::empty());
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
        let mut resampler = audio::StreamingResampler::new(capture_rate, audio::TARGET_SAMPLE_RATE);
        let mut vad_buf: Vec<f32> = Vec::new();

        // Preroll buffer: keeps the last 200ms of audio so speech onset isn't clipped
        let mut preroll: VecDeque<f32> = VecDeque::new();
        let preroll_samples = audio::PREROLL_SAMPLES;

        // Accumulate all transcriptions from this session
        let mut session = SessionTranscript::default();

        // Recording loop — runs until state changes
        loop {
            let current = state.load(Ordering::SeqCst);
            if current == STATE_STOPPING || current == STATE_CANCELLING || current == STATE_SHUTDOWN
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
                audio::push_preroll(&mut preroll, &vad_chunk, preroll_samples);

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
                        match transcribe_buffer(&mut model, &mel_config, &mut utterance_buffer) {
                            Ok(Some(utterance)) => {
                                eprintln!("[daemon] Transcribed: {}", utterance.text);
                                session.append(utterance);

                                // In capture mode, exit after first utterance
                                if is_capture_mode {
                                    break;
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                eprintln!("[daemon] Transcription error: {e}");
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
            if is_capture_mode && !session.is_empty() {
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
                    let _ = tx.send(CaptureResult::empty());
                }
            }

            drop(capture);
            state.store(STATE_IDLE, Ordering::SeqCst);
            eprintln!("[daemon] Session cancelled. Ready for next command.");
            continue;
        }

        // ── Flush remaining audio (non-cancel path) ─────────────────
        if !is_capture_mode || session.is_empty() {
            let mut remaining = resampler.finish();
            if !remaining.is_empty() {
                vad_buf.append(&mut remaining);
            }
            while vad_buf.len() >= VAD_CHUNK_SAMPLES {
                let vad_chunk: Vec<f32> = vad_buf.drain(..VAD_CHUNK_SAMPLES).collect();
                let speech_prob = vad_model.process_chunk(&vad_chunk)?;
                audio::push_preroll(&mut preroll, &vad_chunk, preroll_samples);
                let event = segmenter.process(speech_prob);

                match event {
                    VadEvent::SpeechStart => {
                        utterance_buffer.clear();
                        let preroll_vec: Vec<f32> = preroll.iter().copied().collect();
                        utterance_buffer.push(&preroll_vec);
                        preroll.clear();
                    }
                    VadEvent::SpeechEnd => {
                        match transcribe_buffer(&mut model, &mel_config, &mut utterance_buffer) {
                            Ok(Some(utterance)) => {
                                eprintln!("[daemon] Transcribed: {}", utterance.text);
                                session.append(utterance);
                            }
                            Ok(None) => {}
                            Err(e) => {
                                eprintln!("[daemon] Transcription error: {e}");
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
                match transcribe_buffer(&mut model, &mel_config, &mut utterance_buffer) {
                    Ok(Some(utterance)) => {
                        eprintln!("[daemon] Transcribed (final): {}", utterance.text);
                        session.append(utterance);
                    }
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("[daemon] Final transcription error: {e}");
                    }
                }
            }
        }

        // ── End of recording session ────────────────────────────────

        // If capture mode, send result through the channel
        if is_capture_mode {
            if let Some(tx) = capture_tx.lock().await.take() {
                let _ = tx.send(session.capture_result());
            }
        }

        // Output result to stdout (for non-capture mode, or both)
        if !session.is_empty() {
            if !is_capture_mode {
                println!("{}", session.text);
            }

            if clipboard {
                match copy_text(&session.text) {
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
