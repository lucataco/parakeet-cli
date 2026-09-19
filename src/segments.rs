pub const FRAME_SAMPLES: usize = 1280;
pub const CORE_SAMPLES: usize = FRAME_SAMPLES * 320;
pub const CONTEXT_SAMPLES: usize = FRAME_SAMPLES * 8;

/// How much new audio must arrive between two interim previews (0.75 s).
pub const PREVIEW_INTERVAL_SAMPLES: usize = 12_000;
/// Audio that must be pending before the first preview is worth running (0.4 s).
pub const PREVIEW_MIN_SAMPLES: usize = 6_400;
/// Longest owned span a preview re-encodes (15 s). Beyond this the preview
/// covers only the newest audio and is reported as truncated, which keeps the
/// per-tick cost bounded however long the utterance grows.
pub const PREVIEW_MAX_OWNED_SAMPLES: usize = 16_000 * 15;
/// Silence appended to every preview (0.3 s). Audio cut mid-phoneme makes the
/// decoder invent a tail ("open the notes and I'm not going to be able to do
/// it" for 0.75 s of "open the notes app"); a short silent run-out lets it
/// finish the last word instead. Measured: 0.3 s removes the effect entirely
/// and more gives no further benefit.
pub const PREVIEW_TAIL_SILENCE_SAMPLES: usize = 4_800;

#[derive(Debug)]
pub struct Segment {
    pub samples: Vec<f32>,
    pub owned_start: usize,
    pub owned_end: usize,
}

/// A best-effort look at audio that has not yet been committed to a segment.
/// Interim text derived from it is advisory; the final transcript comes only
/// from committed segments.
#[derive(Debug)]
pub struct Preview {
    pub segment: Segment,
    /// True when older pending audio was left out to bound the encode cost, so
    /// the preview's text may be missing a stretch before its first word.
    pub truncated: bool,
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

    /// Samples held back that no committed segment owns yet.
    pub fn owned_pending(&self) -> usize {
        self.pending.len().saturating_sub(self.left_context)
    }

    /// Copies the uncommitted audio for an interim decode without changing
    /// segmentation. The newest `max_owned` owned samples are kept, preceded by
    /// up to one context span so the encoder sees the same left context a
    /// committed segment would. Returns `None` when nothing is pending.
    pub fn preview(&self, max_owned: usize) -> Option<Preview> {
        let owned = self.owned_pending();
        if owned == 0 {
            return None;
        }
        let (window_start, owned_start) = if owned <= max_owned {
            (0, self.left_context)
        } else {
            let owned_start_index = self.pending.len() - max_owned;
            let context = CONTEXT_SAMPLES.min(owned_start_index);
            (owned_start_index - context, context)
        };
        let mut samples = self.pending[window_start..].to_vec();
        samples.extend(std::iter::repeat_n(0.0, PREVIEW_TAIL_SILENCE_SAMPLES));
        Some(Preview {
            segment: Segment {
                owned_start,
                owned_end: samples.len(),
                samples,
            },
            truncated: owned > max_owned,
        })
    }
}

/// Decides when enough new audio has accumulated for another interim preview.
/// Sample counts rather than wall-clock time keep the policy deterministic.
#[derive(Debug, Default)]
pub struct PreviewCadence {
    since_last: usize,
}

impl PreviewCadence {
    /// Records `new_samples` of arriving audio and returns true when a preview
    /// should be produced now, given `owned_pending` uncommitted samples.
    pub fn observe(&mut self, new_samples: usize, owned_pending: usize) -> bool {
        self.since_last += new_samples;
        if self.since_last >= PREVIEW_INTERVAL_SAMPLES && owned_pending >= PREVIEW_MIN_SAMPLES {
            self.since_last = 0;
            true
        } else {
            false
        }
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

    #[test]
    fn preview_covers_all_pending_audio_and_leaves_segmentation_untouched() {
        let mut segmenter = Segmenter::default();
        assert!(segmenter.preview(PREVIEW_MAX_OWNED_SAMPLES).is_none());
        let audio: Vec<f32> = (0..24_000).map(|i| i as f32).collect();
        assert!(segmenter.push(&audio).is_empty());
        let preview = segmenter.preview(PREVIEW_MAX_OWNED_SAMPLES).unwrap();
        assert!(!preview.truncated);
        assert_eq!(preview.segment.owned_start, 0);
        assert_eq!(
            preview.segment.owned_end,
            24_000 + PREVIEW_TAIL_SILENCE_SAMPLES,
            "the silent run-out is part of the decoded span so the last word can finish"
        );
        assert_eq!(&preview.segment.samples[..24_000], &audio[..]);
        assert!(preview.segment.samples[24_000..].iter().all(|s| *s == 0.0));
        assert_eq!(segmenter.owned_pending(), 24_000);
        // The final segment is exactly what it would have been without the preview.
        let tail = segmenter.finish().unwrap();
        assert_eq!(tail.samples, audio);
        assert_eq!((tail.owned_start, tail.owned_end), (0, 24_000));
    }

    #[test]
    fn preview_after_a_committed_segment_owns_only_the_new_tail() {
        let mut segmenter = Segmenter::default();
        let total = CORE_SAMPLES + CONTEXT_SAMPLES + 5_000;
        let audio: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let committed = segmenter.push(&audio);
        assert_eq!(committed.len(), 1);
        let preview = segmenter.preview(PREVIEW_MAX_OWNED_SAMPLES).unwrap();
        assert!(!preview.truncated);
        assert_eq!(preview.segment.owned_start, CONTEXT_SAMPLES);
        assert_eq!(
            segmenter.owned_pending(),
            CONTEXT_SAMPLES + 5_000,
            "the encoder context after the commit plus the new audio"
        );
        assert_eq!(
            preview.segment.samples[preview.segment.owned_start], CORE_SAMPLES as f32,
            "owned preview audio starts with the first sample the committed segment did not own"
        );
        assert_eq!(
            preview.segment.samples[0],
            (CORE_SAMPLES - CONTEXT_SAMPLES) as f32,
            "preceded by the same left context a committed segment would see"
        );
    }

    #[test]
    fn long_preview_is_capped_to_the_newest_audio_with_context() {
        let mut segmenter = Segmenter::default();
        let total = PREVIEW_MAX_OWNED_SAMPLES + 40_000;
        let audio: Vec<f32> = (0..total).map(|i| i as f32).collect();
        assert!(
            segmenter.push(&audio).is_empty(),
            "still below one core span"
        );
        let preview = segmenter.preview(PREVIEW_MAX_OWNED_SAMPLES).unwrap();
        assert!(preview.truncated);
        assert_eq!(preview.segment.owned_start, CONTEXT_SAMPLES);
        assert_eq!(
            preview.segment.owned_end - preview.segment.owned_start,
            PREVIEW_MAX_OWNED_SAMPLES + PREVIEW_TAIL_SILENCE_SAMPLES
        );
        let last_audio = preview.segment.samples.len() - PREVIEW_TAIL_SILENCE_SAMPLES - 1;
        assert_eq!(preview.segment.samples[last_audio], (total - 1) as f32);
        assert_eq!(
            preview.segment.samples[preview.segment.owned_start],
            (total - PREVIEW_MAX_OWNED_SAMPLES) as f32
        );

        // With little audio before the window, the context shrinks to what exists.
        let mut short = Segmenter::default();
        short.push(&audio[..PREVIEW_MAX_OWNED_SAMPLES + 100]);
        let preview = short.preview(PREVIEW_MAX_OWNED_SAMPLES).unwrap();
        assert!(preview.truncated);
        assert_eq!(preview.segment.owned_start, 100);
    }

    #[test]
    fn cadence_waits_for_interval_and_minimum_audio() {
        let mut cadence = PreviewCadence::default();
        assert!(!cadence.observe(PREVIEW_INTERVAL_SAMPLES, PREVIEW_MIN_SAMPLES - 1));
        assert!(
            cadence.observe(0, PREVIEW_MIN_SAMPLES),
            "the interval was already reached; only the minimum was missing"
        );
        assert!(!cadence.observe(PREVIEW_INTERVAL_SAMPLES - 1, 100_000));
        assert!(cadence.observe(1, 100_000));
        // The count restarts from zero at each tick (no carry), so with 100 ms
        // chunks a tick lands on every eighth chunk.
        let mut ticks = 0;
        for _ in 0..100 {
            if cadence.observe(1_600, 100_000) {
                ticks += 1;
            }
        }
        let chunks_per_tick = PREVIEW_INTERVAL_SAMPLES.div_ceil(1_600);
        assert_eq!(chunks_per_tick, 8);
        assert_eq!(ticks, 100 / chunks_per_tick);
    }
}
