pub mod audio;
pub mod cli;
pub mod clipboard;
pub mod download;
mod integrity;
pub mod listen;
pub mod model;
pub mod segments;
pub mod serve;
pub mod session_capture;
pub mod session_recognition;
pub mod vad;

pub const DAEMON_PROTOCOL_VERSION: u32 = 1;
