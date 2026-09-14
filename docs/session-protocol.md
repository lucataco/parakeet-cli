# Daemon protocol 1 (v0.1.6)

`parakeet protocol-version` prints `1`. `serve` now writes UTF-8 NDJSON events to
stdout, with human-readable diagnostics on stderr. Plain-transcript consumers
must read the `text` field of `complete` events. Newlines inside text are escaped.

Send one newline-terminated command per Unix socket connection:

```json
{"command":"start","session_id":"a-client-generated-unique-id"}
{"command":"stop","session_id":"a-client-generated-unique-id"}
```

Replies include `status`, `state`, `session_id`, and `protocol_version`. `start`
acknowledges only after the microphone stream starts. A busy engine rejects a
second start. `stop` acknowledges the request while the engine remains busy;
only `complete` ends the session. Supplied IDs must match for stop/cancel.

```json
{"type":"session_started","session_id":"…"}
{"type":"transcribing","session_id":"…"}
{"type":"complete","session_id":"…","status":"ok","text":"First line.\nSecond line.","failed_segments":0,"dropped_samples":0,"duration":12.0,"inference_time":0.8,"message":null}
```

Completion status is `ok`, `partial`, `empty`, or `error`. Empty sessions and
capture-start failures also complete. Loss with no recovered text is `error`,
not `empty`. `failed_segments` includes inference/VAD failures and panics;
`dropped_samples` is the 16 kHz-equivalent count of callback or recognition-queue
overruns. A successful engine result cannot detect every acoustic misrecognition.

`status`, `cancel`, `shutdown`, bare commands, `toggle`, SIGUSR1 (toggle), and
SIGUSR2 (stop) remain available. Legacy clients may omit IDs; the engine generates
one at start. `capture` waits for a speech endpoint or thirty seconds, then returns
the result on its socket, including `transcript_status` and loss counters.

## Audio pipeline

The protocol-1 daemon lives in `src/serve.rs`. `DaemonContext` owns shared control
state; `src/serve/protocol.rs`, `collection.rs`, `runtime.rs` and `worker.rs` handle
wire messages, audio collection, runtime files and inference delivery. The wire
version is defined once by `DAEMON_PROTOCOL_VERSION` in `src/lib.rs`. CPAL
capture and collection stay on their own thread. The callback never waits for
recognition. It counts failed queue sends. Stop freezes callback ingress before
draining queued samples and flushing the streaming resampler's fractional tail.

The collector emits 25.6-second owned spans plus 0.64 seconds of encoder context
on each side. Every accepted sample has exactly one owner; no sliding window
evicts old speech. Both encoder and decoder see overlapping boundary context;
only tokens positioned in the owned encoder-frame span are emitted. Token sequences are joined
without text-based deduplication, so deliberate repeated answers are preserved
whenever recognized. Queues have bounded memory and report any overrun.

`parakeet transcribe recording.wav --session --format json` replays the same
segmenter/recognizer and tail finalization. It does not exercise a physical input
device. Superkeet's `Tests/AudioRegression` contains prompts, synthetic-fixture
generation, exact critical-word/count gates, WER checks and provenance.

Validation:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --locked
cargo build --release --locked --bin parakeet
```

The original `listen` command keeps its existing API; the new bounded pipeline
is used by `serve` and `transcribe --session`.
