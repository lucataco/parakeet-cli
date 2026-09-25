# Daemon protocol 2 (v0.1.7)

`parakeet protocol-version` prints `2`. `serve` writes UTF-8 NDJSON events to
stdout, with human-readable diagnostics on stderr. Plain-transcript consumers
must read the `text` field of `complete` events. Newlines inside text are escaped.

Protocol 2 is a superset of protocol 1: a client that sends protocol-1 commands
receives exactly the protocol-1 event stream. The one addition is opt-in interim
text (`partial` events) for a session that asks for it.

Send one newline-terminated command per Unix socket connection:

```json
{"command":"start","session_id":"a-client-generated-unique-id"}
{"command":"start","session_id":"…","partials":true}
{"command":"stop","session_id":"a-client-generated-unique-id"}
```

Replies include `status`, `state`, `session_id`, and `protocol_version`. `start`
acknowledges only after the microphone stream starts. A busy engine rejects a
second start. `stop` acknowledges the request while the engine remains busy;
only `complete` ends the session. Supplied IDs must match for stop/cancel.

```json
{"type":"session_started","session_id":"…"}
{"type":"partial","session_id":"…","text":"Open the notes up","sequence":1,"truncated":false}
{"type":"partial","session_id":"…","text":"Open the notes app and create a","sequence":2,"truncated":false}
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
`capture` never streams partials.

## Interim text (`partial`)

When `start` carries `"partials": true`, the engine streams its current best
reading of the recording roughly every 0.5 s of captured audio (0.75 s before
v0.1.9), once at least
0.4 s is pending. Partials are **advisory**: they are decoded from audio that no
committed segment owns yet, and the final `complete` text comes only from the
committed pipeline described below, which partials never alter. A client must
treat the `complete` text as authoritative and may see a partial's last word
change or disappear in the next one.

- `text` is the running transcript: tokens from every committed segment plus a
  fresh decode of the uncommitted tail. Each partial replaces the previous one.
- `sequence` starts at 1 and increases by one per emitted partial within a
  session. Unchanged text is not repeated.
- `audio_ms` (v0.1.9+) is how much of the recording the preview had heard, in
  milliseconds from `start`. It lines interim text up with the audio; clients
  that don't need it can ignore it.
- `truncated` is true when the uncommitted tail exceeded 15 s and only its
  newest 15 s were decoded, so `text` may skip a stretch between its committed
  prefix and its newest words. Below 15 s it is always false.
- Every preview decode appends 0.3 s of silence to the audio before encoding.
  Without it the TDT decoder invents a tail for audio cut mid-phoneme
  ("open the notes and I'm not going to be able to do it" for 0.75 s of
  "open the notes app"); with it the same audio reads "Open the notes up".
- Partials stop as soon as `stop` or `cancel` is received; `transcribing` and
  `complete` follow as before. All three event kinds are written by the same
  thread, so a `partial` can never arrive after its session's `complete`.
- Previews travel on a separate single-slot queue and are skipped, never
  queued, when the recognizer is busy. They cannot displace a committed segment
  and are never counted in `dropped_samples` or `failed_segments`.

Cost: each partial re-encodes the uncommitted tail (about 30 ms plus 14 ms per
second of audio on Apple Silicon CPU), so a 10 s utterance costs ~150 ms per
tick and the 15 s cap bounds it at ~250 ms. Sessions without `partials` do no
extra work.

`parakeet transcribe recording.wav --session --partials --format json` replays a
file through the same cadence and preview code and prints the `partial` lines
before the `complete` line, so interim output can be checked without a
microphone.

## Audio pipeline

The daemon lives in `src/serve.rs`. `DaemonContext` owns shared control state;
`src/serve/protocol.rs`, `collection.rs`, `runtime.rs` and `worker.rs` handle
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
`Segmenter::preview` copies the uncommitted tail for interim decodes without
changing segmentation; `PreviewCadence` decides when.

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

The original `listen` command keeps its existing API; the bounded pipeline is
used by `serve` and `transcribe --session`.
