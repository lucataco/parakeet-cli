use super::TARGET_SAMPLE_RATE;

pub struct AudioBuffer {
    samples: Vec<f32>,
    max_samples: usize,
}

impl AudioBuffer {
    pub fn new(max_duration_secs: f32) -> Self {
        let max_samples = (max_duration_secs * TARGET_SAMPLE_RATE as f32) as usize;
        Self {
            samples: Vec::with_capacity(TARGET_SAMPLE_RATE as usize),
            max_samples,
        }
    }

    pub fn push(&mut self, new_samples: &[f32]) {
        self.samples.extend_from_slice(new_samples);

        if self.samples.len() > self.max_samples {
            let excess = self.samples.len() - self.max_samples;
            self.samples.drain(..excess);
        }
    }

    pub fn drain(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.samples)
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / TARGET_SAMPLE_RATE as f32
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
        let mut buf = AudioBuffer::new(0.001);
        buf.push(&vec![1.0; 100]);
        assert!(buf.len() <= 16);
    }

    #[test]
    fn test_duration() {
        let mut buf = AudioBuffer::new(30.0);
        buf.push(&vec![0.0; TARGET_SAMPLE_RATE as usize]);
        assert!((buf.duration_secs() - 1.0).abs() < 0.001);
    }
}
