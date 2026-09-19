use crate::{DAEMON_PROTOCOL_VERSION, session_recognition::SessionResult};
use anyhow::Result;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{io::Write, time::Duration};
use tokio::io::{AsyncRead, AsyncReadExt};

#[derive(Deserialize, Debug, PartialEq)]
pub(super) struct Command {
    pub command: String,
    pub session_id: Option<String>,
    /// Protocol 2: ask `start` to stream interim `partial` events while the
    /// session records. Absent or false keeps protocol-1 output exactly.
    #[serde(default)]
    pub partials: bool,
}

impl Command {
    pub fn bare(command: &str) -> Self {
        Self {
            command: command.into(),
            session_id: None,
            partials: false,
        }
    }
}

pub(super) fn parse_command(bytes: &[u8]) -> serde_json::Result<Command> {
    if bytes.first() == Some(&b'{') {
        serde_json::from_slice(bytes)
    } else {
        Ok(Command::bare(String::from_utf8_lossy(bytes).trim()))
    }
}

pub(super) async fn read_command(stream: &mut (impl AsyncRead + Unpin)) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    loop {
        let byte = match tokio::time::timeout(Duration::from_secs(5), stream.read_u8()).await? {
            Ok(byte) => byte,
            Err(error)
                if error.kind() == std::io::ErrorKind::UnexpectedEof && !bytes.is_empty() =>
            {
                break;
            }
            Err(error) => return Err(error.into()),
        };
        if byte == b'\n' {
            break;
        }
        anyhow::ensure!(bytes.len() < 4096, "Command is too large");
        bytes.push(byte);
    }
    Ok(bytes)
}

pub(super) fn acknowledged(state: &str, session_id: Option<&str>) -> Value {
    json!({"status": "ok", "state": state, "session_id": session_id, "protocol_version": DAEMON_PROTOCOL_VERSION})
}

pub(super) fn error(message: impl std::fmt::Display) -> Value {
    json!({"status": "error", "message": message.to_string(), "protocol_version": DAEMON_PROTOCOL_VERSION})
}

pub(super) fn session_started(session_id: &str) -> Value {
    json!({"type": "session_started", "session_id": session_id})
}

pub(super) fn transcribing(session_id: &str) -> Value {
    json!({"type": "transcribing", "session_id": session_id})
}

/// Interim text for a session that is still recording. `sequence` increases by
/// one per emitted partial so consumers can drop stale arrivals; `truncated`
/// means the preview window left out older uncommitted audio, so `text` may
/// skip a stretch between its committed prefix and its newest words.
pub(crate) fn partial(session_id: &str, text: &str, sequence: u64, truncated: bool) -> Value {
    json!({"type": "partial", "session_id": session_id, "text": text,
        "sequence": sequence, "truncated": truncated})
}

/// Turns preview decodes into `partial` events, numbering them and skipping
/// repeats so an unchanged transcript never produces a new line.
#[derive(Default)]
pub(crate) struct PartialEmitter {
    sequence: u64,
    last: Option<(String, bool)>,
}

impl PartialEmitter {
    pub fn next(&mut self, session_id: &str, text: String, truncated: bool) -> Option<Value> {
        if text.trim().is_empty() || self.last.as_ref() == Some(&(text.clone(), truncated)) {
            return None;
        }
        self.sequence += 1;
        let event = partial(session_id, &text, self.sequence, truncated);
        self.last = Some((text, truncated));
        Some(event)
    }
}

pub(crate) fn completion(result: &SessionResult, session_id: &str) -> Value {
    let failed =
        result.failed_segments > 0 || result.dropped_samples > 0 || result.message.is_some();
    let status = match (result.text.is_empty(), failed) {
        (false, false) => "ok",
        (false, true) => "partial",
        (true, false) => "empty",
        (true, true) => "error",
    };
    json!({"type": "complete", "session_id": session_id, "status": status,
        "text": result.text, "failed_segments": result.failed_segments,
        "dropped_samples": result.dropped_samples, "message": result.message,
        "duration": result.duration, "inference_time": result.inference_time})
}

pub(super) fn capture_reply(result: &SessionResult, session_id: &str) -> Value {
    let mut value = completion(result, session_id);
    value["transcript_status"] = value["status"].clone();
    value["status"] = json!(if result.text.is_empty() && result.message.is_some() {
        "error"
    } else {
        "ok"
    });
    value["state"] = json!("idle");
    value
}

pub(super) fn emit(value: Value) {
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    if writeln!(output, "{value}")
        .and_then(|_| output.flush())
        .is_err()
    {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    #[test]
    fn completion_status_and_capture_transport_status_are_distinct() {
        for (text, failed_segments, dropped_samples, message, expected) in [
            ("text", 0, 0, None, "ok"),
            ("", 0, 0, None, "empty"),
            ("text", 1, 0, None, "partial"),
            ("text", 0, 1, None, "partial"),
            ("", 1, 0, None, "error"),
            ("", 0, 1, None, "error"),
            ("", 0, 0, Some("capture failed"), "error"),
        ] {
            let result = SessionResult {
                text: text.into(),
                failed_segments,
                dropped_samples,
                message: message.map(str::to_owned),
                ..Default::default()
            };
            let event = completion(&result, "id");
            assert_eq!(event["status"], expected);
            assert_eq!(event["session_id"], "id");
            assert!(event.get("message").is_some());
            let reply = capture_reply(&result, "id");
            assert_eq!(reply["transcript_status"], expected);
            assert_eq!(
                reply["status"],
                if text.is_empty() && message.is_some() {
                    "error"
                } else {
                    "ok"
                }
            );
            assert_eq!(reply["state"], "idle");
        }
        assert_eq!(
            acknowledged("idle", None),
            json!({"status":"ok", "state":"idle", "session_id":null, "protocol_version":DAEMON_PROTOCOL_VERSION})
        );
        assert!(error("invalid").get("session_id").is_none());
    }

    #[tokio::test]
    async fn fragmented_unicode_and_eof_commands_preserve_framing() {
        let wire = "{\"command\":\"start\",\"session_id\":\"🦜中文\"}\n".as_bytes();
        let (mut writer, mut reader) = tokio::io::duplex(8);
        let send = async {
            for byte in wire {
                writer.write_all(&[*byte]).await.unwrap();
            }
        };
        let receive = read_command(&mut reader);
        let (_, result) = tokio::join!(send, receive);
        assert_eq!(
            parse_command(&result.unwrap())
                .unwrap()
                .session_id
                .as_deref(),
            Some("🦜中文")
        );
        let mut bare = &b"  status  "[..];
        assert_eq!(
            parse_command(&read_command(&mut bare).await.unwrap()).unwrap(),
            Command::bare("status")
        );
        assert!(parse_command(b"{invalid}").is_err());
        assert_eq!(parse_command(b" STATUS ").unwrap().command, "STATUS");
    }

    #[test]
    fn partials_are_opt_in_per_start_command() {
        let plain = parse_command(br#"{"command":"start","session_id":"a"}"#).unwrap();
        assert!(
            !plain.partials,
            "protocol-1 clients never asked and never receive partials"
        );
        let wanted =
            parse_command(br#"{"command":"start","session_id":"a","partials":true}"#).unwrap();
        assert!(wanted.partials);
        let declined = parse_command(br#"{"command":"start","partials":false}"#).unwrap();
        assert!(!declined.partials);
        assert!(!Command::bare("start").partials);
    }

    #[test]
    fn partial_events_are_numbered_deduplicated_and_never_empty() {
        let mut emitter = PartialEmitter::default();
        assert!(emitter.next("s", "".into(), false).is_none());
        assert!(emitter.next("s", "   ".into(), false).is_none());
        let first = emitter.next("s", "open the".into(), false).unwrap();
        assert_eq!(first["type"], "partial");
        assert_eq!(first["session_id"], "s");
        assert_eq!(first["text"], "open the");
        assert_eq!(first["sequence"], 1);
        assert_eq!(first["truncated"], false);
        assert!(
            emitter.next("s", "open the".into(), false).is_none(),
            "unchanged text is not repeated"
        );
        let second = emitter.next("s", "open the notes".into(), false).unwrap();
        assert_eq!(second["sequence"], 2);
        let flagged = emitter.next("s", "open the notes".into(), true).unwrap();
        assert_eq!(flagged["sequence"], 3);
        assert_eq!(
            flagged["truncated"], true,
            "a change in the window flag is worth reporting"
        );
        assert!(
            !partial("s", "first\nsecond", 1, false)
                .to_string()
                .contains('\n')
        );
    }

    #[tokio::test]
    async fn command_limit_accepts_exactly_4096_bytes_and_rejects_more() {
        let allowed = vec![b'x'; 4096];
        assert_eq!(
            read_command(&mut allowed.as_slice()).await.unwrap().len(),
            4096
        );
        let oversized = vec![b'x'; 4097];
        assert!(read_command(&mut oversized.as_slice()).await.is_err());
        assert!(read_command(&mut &b""[..]).await.is_err());
    }
}
