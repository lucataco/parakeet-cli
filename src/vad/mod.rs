pub mod silero;

pub use silero::{
    SileroVad, VAD_CHUNK_SAMPLES, VadEvent, VadSegmenter, VadState, ensure_vad_model,
};
