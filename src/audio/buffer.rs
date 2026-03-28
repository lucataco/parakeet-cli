/// Ring buffer for accumulating streaming audio samples.
///
/// Used in the listen pipeline to collect audio between VAD
/// speech-start and speech-end events, then drain the complete
/// utterance for transcription.

/// A growable audio buffer that accumulates samples for a single utterance.
///
/// Unlike a fixed-size ring buffer, this grows as speech continues
/// and is drained completely when the utterance ends. This is simpler
/// and more appropriate for VAD-segmented transcription where we need
/// the full utterance audio.
pub struct AudioBuffer {
    /// Accumulated mono f32 samples at 16kHz.
    samples: Vec<f32>,
    /// Maximum duration in samples (safety limit to prevent OOM).
    max_samples: usize,
}

impl AudioBuffer {
    /// Create a new audio buffer.
    ///
    /// # Arguments
    /// * `max_duration_secs` - Maximum utterance duration in seconds (safety limit).
    pub fn new(max_duration_secs: f32) -> Self {
        let max_samples = (max_duration_secs * 16000.0) as usize;
        Self {
            samples: Vec::with_capacity(16000), // pre-allocate 1 second
            max_samples,
        }
    }

    /// Append samples to the buffer.
    ///
    /// If the buffer would exceed max_samples, the oldest samples are
    /// dropped (sliding window behavior as a safety measure).
    pub fn push(&mut self, new_samples: &[f32]) {
        self.samples.extend_from_slice(new_samples);

        // Safety: if we exceed max, keep only the most recent max_samples
        if self.samples.len() > self.max_samples {
            let excess = self.samples.len() - self.max_samples;
            self.samples.drain(..excess);
        }
    }

    /// Drain all samples from the buffer, returning them.
    /// The buffer is left empty and ready for the next utterance.
    pub fn drain(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.samples)
    }

    /// Clear the buffer without returning samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Current number of samples in the buffer.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the buffer is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Duration of buffered audio in seconds (at 16kHz).
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / 16000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_drain() {
        let mut buf = AudioBuffer::new(30.0);
        buf.push(&[1.0, 2.0, 3.0]);
        buf.push(&[4.0, 5.0]);
        assert_eq!(buf.len(), 5);

        let data = buf.drain();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_max_samples_safety() {
        // 0.001 seconds = 16 samples at 16kHz
        let mut buf = AudioBuffer::new(0.001);
        buf.push(&vec![1.0; 100]);
        // Should be capped at 16 samples
        assert!(buf.len() <= 16);
    }

    #[test]
    fn test_duration() {
        let mut buf = AudioBuffer::new(30.0);
        buf.push(&vec![0.0; 16000]);
        assert!((buf.duration_secs() - 1.0).abs() < 0.001);
    }
}
