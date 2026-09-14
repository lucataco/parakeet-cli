pub const FRAME_SAMPLES: usize = 1280;
pub const CORE_SAMPLES: usize = FRAME_SAMPLES * 320;
pub const CONTEXT_SAMPLES: usize = FRAME_SAMPLES * 8;

#[derive(Debug)]
pub struct Segment {
    pub samples: Vec<f32>,
    pub owned_start: usize,
    pub owned_end: usize,
}

#[derive(Default)]
pub struct Segmenter {
    pending: Vec<f32>,
    left_context: usize,
}

impl Segmenter {
    pub fn push(&mut self, samples: &[f32]) -> Vec<Segment> {
        self.pending.extend_from_slice(samples);
        let mut ready = Vec::new();
        while self.pending.len() >= self.left_context + CORE_SAMPLES + CONTEXT_SAMPLES {
            let end = self.left_context + CORE_SAMPLES;
            ready.push(Segment {
                samples: self.pending[..end + CONTEXT_SAMPLES].to_vec(),
                owned_start: self.left_context,
                owned_end: end,
            });
            self.pending.drain(..end - CONTEXT_SAMPLES);
            self.left_context = CONTEXT_SAMPLES;
        }
        ready
    }

    pub fn finish(&mut self) -> Option<Segment> {
        let samples = std::mem::take(&mut self.pending);
        let start = std::mem::take(&mut self.left_context);
        (samples.len() > start).then_some(Segment {
            owned_start: start,
            owned_end: samples.len(),
            samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_minutes_preserves_every_sample_in_order_exactly_once() {
        let mut segmenter = Segmenter::default();
        let count = 16000 * 600;
        let mut next = 0;
        for offset in (0..count).step_by(997) {
            let chunk: Vec<f32> = (offset..(offset + 997).min(count))
                .map(|i| i as f32)
                .collect();
            for segment in segmenter.push(&chunk) {
                assert!(segment.samples.len() <= CORE_SAMPLES + 2 * CONTEXT_SAMPLES);
                for sample in &segment.samples[segment.owned_start..segment.owned_end] {
                    assert_eq!(*sample, next as f32);
                    next += 1;
                }
            }
        }
        let segment = segmenter.finish().unwrap();
        for sample in &segment.samples[segment.owned_start..segment.owned_end] {
            assert_eq!(*sample, next as f32);
            next += 1;
        }
        assert_eq!(next, count);
        assert!(segmenter.finish().is_none());
    }

    #[test]
    fn immediate_stop_keeps_sub_vad_frame_tail() {
        let mut segmenter = Segmenter::default();
        assert!(segmenter.push(&[0.2; 501]).is_empty());
        assert_eq!(segmenter.finish().unwrap().samples.len(), 501);
    }
}
