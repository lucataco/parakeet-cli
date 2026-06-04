/// Live streaming transcription pipeline.
///
/// Pipeline: mic capture -> resample to 16kHz -> VAD segmentation -> transcribe utterances
///
/// The listen loop runs on the main thread:
/// 1. Audio callback pushes chunks through a crossbeam channel
/// 2. Main thread reads chunks, feeds them to the resampler, then to VAD
/// 3. When VAD detects speech, audio is accumulated in a buffer
/// 4. When VAD detects end of speech, the buffered audio is transcribed
/// 5. Transcription result is printed to stdout
use anyhow::Result;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::audio::{self, AudioBuffer};
use crate::clipboard::copy_text;
use crate::model::ParakeetModel;
use crate::vad::{self, SileroVad, VAD_CHUNK_SAMPLES, VadEvent, VadSegmenter, VadState};

const MAX_UTTERANCE_SECS: f32 = 60.0;

/// Compute RMS (root mean square) of a sample buffer.
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn take_pending_audio_on_shutdown(utterance_buffer: &mut AudioBuffer) -> Option<Vec<f32>> {
    if utterance_buffer.duration_secs() > 0.1 {
        Some(utterance_buffer.drain())
    } else {
        utterance_buffer.clear();
        None
    }
}

pub struct ListenConfig<'a> {
    pub device: &'a Option<String>,
    pub model_dir: &'a Path,
    pub vad_threshold: f32,
    pub silence_ms: u64,
    pub clipboard: bool,
    pub debug: bool,
    pub verbose: bool,
    pub use_coreml: bool,
    pub single_utterance: bool,
}

/// Run the live listen pipeline.
///
/// This blocks until Ctrl-C is pressed.
pub async fn run_listen(config: ListenConfig<'_>) -> Result<()> {
    let ListenConfig {
        device,
        model_dir,
        vad_threshold,
        silence_ms,
        clipboard,
        debug,
        verbose,
        use_coreml,
        single_utterance,
    } = config;

    // Download VAD model if needed
    let vad_path = vad::ensure_vad_model(model_dir).await?;

    // Load Parakeet model
    if verbose {
        eprintln!("Loading Parakeet model...");
    }
    let mut model = ParakeetModel::load(model_dir, use_coreml, verbose)?;
    if verbose {
        eprintln!();
    }

    // Load Silero VAD
    let mut vad_model = SileroVad::load(&vad_path, verbose)?;
    let mut segmenter = VadSegmenter::new(vad_threshold, silence_ms);
    if verbose {
        eprintln!();
    }

    // Start audio capture
    let capture = audio::start_capture(device)?;
    let capture_rate = capture.sample_rate;
    eprintln!();

    eprintln!("Listening... (press Ctrl-C to stop)");
    eprintln!(
        "VAD threshold: {}, silence timeout: {}ms",
        vad_threshold, silence_ms
    );
    if debug {
        eprintln!("[debug] Debug mode enabled — printing audio levels and VAD probabilities");
    }
    eprintln!();

    // Set up Ctrl-C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .map_err(|e| anyhow::anyhow!("Failed to set Ctrl-C handler: {e}"))?;

    // Audio buffer for accumulating speech utterances
    let mut utterance_buffer = AudioBuffer::new(MAX_UTTERANCE_SECS);
    let mut resampler = audio::StreamingResampler::new(capture_rate, audio::TARGET_SAMPLE_RATE);

    // VAD processing buffer: accumulate resampled audio, process in 512-sample chunks
    let mut vad_buf: Vec<f32> = Vec::new();
    let mut preroll = VecDeque::new();
    let preroll_samples = audio::PREROLL_SAMPLES;

    let mel_config = audio::MelConfig::default();

    // Debug state
    let mut total_capture_samples: u64 = 0;
    let mut total_vad_chunks: u64 = 0;
    let mut debug_vad_chunk_count: u64 = 0; // counter for periodic debug output
    let mut max_speech_prob: f32 = 0.0; // track max prob between debug prints
    let mut peak_speech_prob: f32 = 0.0;
    let mut warned_silent = false; // one-time warning for silent audio
    let mut warned_low_vad = false;
    let mut capture_rms_accum: f32 = 0.0;
    let mut capture_rms_count: u32 = 0;
    let mut capture_rms_total: f32 = 0.0;
    let mut capture_chunk_total: u64 = 0;

    // Print debug every N VAD chunks (~500ms = ~16 chunks at 32ms each)
    let debug_interval: u64 = 16;

    while running.load(Ordering::SeqCst) {
        // Receive audio chunk with a timeout so we can check the running flag
        let chunk = match capture
            .receiver
            .recv_timeout(std::time::Duration::from_millis(100))
        {
            Ok(chunk) => chunk,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                eprintln!("Audio stream disconnected");
                break;
            }
        };

        // Debug: track capture-level audio
        if debug {
            let chunk_rms = rms(&chunk.samples);
            capture_rms_accum += chunk_rms;
            capture_rms_count += 1;
            capture_rms_total += chunk_rms;
            capture_chunk_total += 1;
            total_capture_samples += chunk.samples.len() as u64;
        }

        // Resample from capture rate to 16kHz using a continuous stream.
        let resampled = resampler.process(&chunk.samples);

        // Debug: check for silent audio early on
        if debug && !warned_silent && total_capture_samples > (capture_rate as u64) {
            // After ~1 second of audio, check if it's silent
            let avg_rms = if capture_rms_count > 0 {
                capture_rms_accum / capture_rms_count as f32
            } else {
                0.0
            };
            if avg_rms < 1e-5 {
                eprintln!();
                eprintln!("[WARNING] Audio appears silent (avg RMS: {:.8})", avg_rms);
                eprintln!(
                    "[WARNING] Check: macOS System Settings > Privacy & Security > Microphone"
                );
                eprintln!("[WARNING] Make sure your terminal app has microphone permission.");
                eprintln!();
            }
            warned_silent = true;
        }

        // Feed resampled audio to VAD buffer
        vad_buf.extend_from_slice(&resampled);

        // Process complete VAD chunks (512 samples = 32ms each)
        while vad_buf.len() >= VAD_CHUNK_SAMPLES {
            let vad_chunk: Vec<f32> = vad_buf.drain(..VAD_CHUNK_SAMPLES).collect();

            // Run VAD
            let speech_prob = vad_model.process_chunk(&vad_chunk)?;

            total_vad_chunks += 1;
            debug_vad_chunk_count += 1;

            // Track max prob for debug interval
            if speech_prob > max_speech_prob {
                max_speech_prob = speech_prob;
            }
            if speech_prob > peak_speech_prob {
                peak_speech_prob = speech_prob;
            }

            // Debug: periodic output every ~500ms
            if debug && debug_vad_chunk_count >= debug_interval {
                let chunk_rms = rms(&vad_chunk);
                let avg_capture_rms = if capture_rms_count > 0 {
                    capture_rms_accum / capture_rms_count as f32
                } else {
                    0.0
                };
                eprintln!(
                    "[debug] vad_prob={:.3} (max={:.3}) | rms_16k={:.5} rms_capture={:.5} | state={:?} | chunks={}",
                    speech_prob,
                    max_speech_prob,
                    chunk_rms,
                    avg_capture_rms,
                    segmenter.state(),
                    total_vad_chunks,
                );
                debug_vad_chunk_count = 0;
                max_speech_prob = 0.0;
                capture_rms_accum = 0.0;
                capture_rms_count = 0;
            }

            if debug
                && !warned_low_vad
                && total_capture_samples > (capture_rate as u64 * 3)
                && capture_chunk_total > 0
            {
                let avg_capture_rms = capture_rms_total / capture_chunk_total as f32;
                if avg_capture_rms > 0.003 && peak_speech_prob < 0.20 {
                    eprintln!();
                    eprintln!(
                        "[WARNING] Audio is non-silent (avg RMS {:.5}) but VAD never exceeded {:.3}",
                        avg_capture_rms, peak_speech_prob
                    );
                    eprintln!(
                        "[WARNING] This usually indicates bad live preprocessing. Try a lower threshold while debugging."
                    );
                    eprintln!();
                    warned_low_vad = true;
                }
            }

            audio::push_preroll(&mut preroll, &vad_chunk, preroll_samples);

            // Run segmenter
            let event = segmenter.process(speech_prob);

            match event {
                VadEvent::SpeechStart => {
                    if debug {
                        eprintln!(
                            "[debug] >>> SpeechStart (prob={:.3}, rms={:.5})",
                            speech_prob,
                            rms(&vad_chunk),
                        );
                    }
                    eprint!("\r[listening] Speech detected...              ");
                    utterance_buffer.clear();
                    let preroll_vec: Vec<f32> = preroll.iter().copied().collect();
                    utterance_buffer.push(&preroll_vec);
                    preroll.clear();
                }
                VadEvent::SpeechEnd => {
                    let duration = utterance_buffer.duration_secs();
                    if debug {
                        eprintln!(
                            "[debug] <<< SpeechEnd (duration={:.2}s, samples={})",
                            duration,
                            (duration * audio::TARGET_SAMPLE_RATE as f32) as usize,
                        );
                    }
                    eprint!("\r[listening] Transcribing {:.1}s utterance...  ", duration);

                    // Transcribe the accumulated utterance
                    let samples = utterance_buffer.drain();

                    if samples.len() > audio::MIN_UTTERANCE_SAMPLES {
                        // At least 0.1s of audio
                        let features = audio::compute_mel_spectrogram(&samples, &mel_config);

                        if debug {
                            eprintln!(
                                "[debug] Mel spectrogram: {} frames x {} bins",
                                features.shape()[0],
                                features.shape()[1],
                            );
                        }

                        match model.transcribe(&features) {
                            Ok(text) => {
                                if !text.trim().is_empty() {
                                    // Clear the status line and print transcription
                                    eprint!("\r                                              \r");
                                    println!("{}", text.trim());

                                    if clipboard {
                                        if let Err(e) = copy_text(text.trim()) {
                                            eprintln!("[clipboard error] {e}");
                                        }
                                    }

                                    // In single-utterance mode, exit after the first
                                    // successful transcription
                                    if single_utterance {
                                        return Ok(());
                                    }
                                } else if debug {
                                    eprintln!("[debug] Transcription returned empty text");
                                }
                            }
                            Err(e) => {
                                eprintln!("\r[error] Transcription failed: {e}");
                            }
                        }
                    } else if debug {
                        eprintln!(
                            "[debug] Utterance too short ({} samples), skipping",
                            samples.len()
                        );
                    }

                    eprint!("\r[listening] Ready...                         ");
                }
                VadEvent::None => {
                    // If we're in the Speaking state, accumulate audio
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
        audio::push_preroll(&mut preroll, &vad_chunk, preroll_samples);
        let event = segmenter.process(speech_prob);
        match event {
            VadEvent::SpeechStart => {
                utterance_buffer.clear();
                let preroll_vec: Vec<f32> = preroll.iter().copied().collect();
                utterance_buffer.push(&preroll_vec);
                preroll.clear();
            }
            VadEvent::SpeechEnd => {}
            VadEvent::None => {
                if segmenter.state() == VadState::Speaking {
                    utterance_buffer.push(&vad_chunk);
                }
            }
        }
    }

    if let Some(samples) = take_pending_audio_on_shutdown(&mut utterance_buffer) {
        let features = audio::compute_mel_spectrogram(&samples, &mel_config);
        match model.transcribe(&features) {
            Ok(text) => {
                let text = text.trim();
                if !text.is_empty() {
                    eprint!("\r                                              \r");
                    println!("{text}");

                    if clipboard {
                        if let Err(e) = copy_text(text) {
                            eprintln!("[clipboard error] {e}");
                        }
                    }
                }
            }
            Err(e) => eprintln!("\r[error] Final transcription failed: {e}"),
        }
    }

    eprintln!();
    eprintln!("Stopped listening.");

    if debug {
        eprintln!(
            "[debug] Total: {} capture samples, {} VAD chunks processed",
            total_capture_samples, total_vad_chunks,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_take_pending_audio_on_shutdown_drains_long_utterance() {
        let mut buf = AudioBuffer::new(1.0);
        buf.push(&vec![0.1; 3200]);

        let drained = take_pending_audio_on_shutdown(&mut buf).unwrap();

        assert_eq!(drained.len(), 3200);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_take_pending_audio_on_shutdown_drops_short_noise() {
        let mut buf = AudioBuffer::new(1.0);
        buf.push(&vec![0.1; 800]);

        assert!(take_pending_audio_on_shutdown(&mut buf).is_none());
        assert!(buf.is_empty());
    }
}
