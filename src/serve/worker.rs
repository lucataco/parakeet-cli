use super::{CANCEL, Job, RUN, State, collection::Collected, protocol};
use crate::{
    segments::Preview,
    session_recognition::{Recognizer, SessionResult},
};
use crossbeam_channel::{Receiver, TryRecvError};
use std::sync::atomic::Ordering;

/// What the worker handles next for a job.
enum Event {
    Collected(Collected),
    Preview(Preview),
    /// The collector stopped offering previews (stop requested or previews
    /// were never enabled); keep waiting for segments only.
    PreviewsClosed,
    Disconnected,
}

/// Committed audio always comes first: a preview is only picked up when no
/// segment is waiting, and the newest queued preview replaces older ones.
fn next_event(job: &Job, previews_open: bool) -> Event {
    match job.receiver.try_recv() {
        Ok(collected) => return Event::Collected(collected),
        Err(TryRecvError::Disconnected) => return Event::Disconnected,
        Err(TryRecvError::Empty) => {}
    }
    if !previews_open {
        return match job.receiver.recv() {
            Ok(collected) => Event::Collected(collected),
            Err(_) => Event::Disconnected,
        };
    }
    crossbeam_channel::select! {
        recv(job.receiver) -> collected => match collected {
            Ok(collected) => Event::Collected(collected),
            Err(_) => Event::Disconnected,
        },
        recv(job.previews) -> preview => match preview {
            Ok(preview) => Event::Preview(job.previews.try_iter().last().unwrap_or(preview)),
            Err(_) => Event::PreviewsClosed,
        },
    }
}

pub(super) fn run(mut recognizer: Recognizer, jobs: Receiver<Job>, state: State, clipboard: bool) {
    for job in jobs {
        let mut result = SessionResult::default();
        let mut finished = false;
        let mut previews_open = true;
        let mut partials = protocol::PartialEmitter::default();
        loop {
            match next_event(&job, previews_open) {
                Event::Collected(Collected::Segment(segment)) => {
                    if job.control.load(Ordering::SeqCst) != CANCEL {
                        recognizer.append(segment, &mut result);
                    }
                }
                Event::Collected(Collected::Done { dropped, error }) => {
                    result.dropped_samples += dropped;
                    if error.is_some() {
                        result.message = error;
                    }
                    finished = true;
                    break;
                }
                Event::Preview(preview) => {
                    // Interim text only while still recording; after stop or
                    // cancel every cycle belongs to the final transcript.
                    if job.control.load(Ordering::SeqCst) == RUN {
                        emit_partial(&mut recognizer, &result, &job.id, preview, &mut partials);
                    }
                }
                Event::PreviewsClosed => previews_open = false,
                Event::Disconnected => break,
            }
        }
        if !finished {
            result.message = Some("Audio collector exited before finalization".into());
        }
        result.text = recognizer.decode(&result.tokens);
        let mut active = state.lock().unwrap_or_else(|error| error.into_inner());
        if active.as_ref().is_some_and(|session| session.id == job.id) {
            *active = None;
            if job.control.load(Ordering::SeqCst) == CANCEL {
                result = SessionResult {
                    message: Some("Recording cancelled".into()),
                    ..Default::default()
                };
            }
            if clipboard && !result.text.is_empty() {
                if let Err(error) = crate::clipboard::copy_text(&result.text) {
                    eprintln!("Clipboard error: {error}");
                }
            }
            if let Some(response) = job.response {
                let _ = response.send(protocol::capture_reply(&result, &job.id));
            }
            protocol::emit(result.completion(&job.id));
        }
    }
}

/// Decodes the committed tokens plus the preview's tokens into one running
/// text. The session's own token list is left untouched.
fn emit_partial(
    recognizer: &mut Recognizer,
    result: &SessionResult,
    session_id: &str,
    preview: Preview,
    partials: &mut protocol::PartialEmitter,
) {
    let truncated = preview.truncated;
    let audio_ms = protocol::samples_to_ms(preview.audio_samples);
    let Some(tokens) = recognizer.preview(preview.segment) else {
        return;
    };
    let mut running = result.tokens.clone();
    running.extend(tokens);
    let text = recognizer.decode(&running);
    if let Some(event) = partials.next(session_id, text, truncated, audio_ms) {
        protocol::emit(event);
    }
}
