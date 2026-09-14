use super::{CANCEL, Job, State, collection::Collected, protocol};
use crate::session_recognition::{Recognizer, SessionResult};
use crossbeam_channel::Receiver;
use std::sync::atomic::Ordering;

pub(super) fn run(mut recognizer: Recognizer, jobs: Receiver<Job>, state: State, clipboard: bool) {
    for job in jobs {
        let mut result = SessionResult::default();
        let mut finished = false;
        for event in job.receiver {
            match event {
                Collected::Segment(segment) => {
                    if job.control.load(Ordering::SeqCst) != CANCEL {
                        recognizer.append(segment, &mut result);
                    }
                }
                Collected::Done { dropped, error } => {
                    result.dropped_samples += dropped;
                    if error.is_some() {
                        result.message = error;
                    }
                    finished = true;
                    break;
                }
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
