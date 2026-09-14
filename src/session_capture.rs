use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

#[derive(Default)]
pub struct CaptureStats {
    pub dropped: AtomicU64,
    pub errors: AtomicU64,
}

pub struct Capture {
    stream: cpal::Stream,
    accepting: Arc<Mutex<bool>>,
    pub receiver: Receiver<Vec<f32>>,
    pub sample_rate: u32,
    pub stats: Arc<CaptureStats>,
}

impl Capture {
    pub fn start(name: &Option<String>) -> Result<Self> {
        let host = cpal::default_host();
        let device = if let Some(name) = name {
            host.input_devices()?
                .find(|device| device.name().is_ok_and(|value| value.contains(name)))
        } else {
            host.default_input_device()
        }
        .context("Audio input device unavailable")?;
        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let (tx, receiver) = crossbeam_channel::bounded(256);
        let stats = Arc::new(CaptureStats::default());
        let accepting = Arc::new(Mutex::new(true));
        macro_rules! build {
            ($type:ty) => {
                build::<$type>(
                    &device,
                    &config.clone().into(),
                    channels,
                    tx,
                    stats.clone(),
                    accepting.clone(),
                )?
            };
        }
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => build!(f32),
            cpal::SampleFormat::F64 => build!(f64),
            cpal::SampleFormat::I8 => build!(i8),
            cpal::SampleFormat::I16 => build!(i16),
            cpal::SampleFormat::I32 => build!(i32),
            cpal::SampleFormat::I64 => build!(i64),
            cpal::SampleFormat::U8 => build!(u8),
            cpal::SampleFormat::U16 => build!(u16),
            cpal::SampleFormat::U32 => build!(u32),
            cpal::SampleFormat::U64 => build!(u64),
            format => anyhow::bail!("Unsupported audio format: {format:?}"),
        };
        stream.play()?;
        Ok(Self {
            stream,
            accepting,
            receiver,
            sample_rate,
            stats,
        })
    }

    pub fn stop(&self) -> Result<()> {
        *self
            .accepting
            .lock()
            .map_err(|_| anyhow::anyhow!("Capture gate failed"))? = false;
        self.stream.pause()?;
        Ok(())
    }
}

pub fn enqueue(tx: &Sender<Vec<f32>>, samples: Vec<f32>, stats: &CaptureStats) {
    let count = samples.len();
    if tx.try_send(samples).is_err() {
        stats.dropped.fetch_add(count as u64, Ordering::Relaxed);
    }
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    tx: Sender<Vec<f32>>,
    stats: Arc<CaptureStats>,
    accepting: Arc<Mutex<bool>>,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + Copy,
    f32: cpal::FromSample<T>,
{
    let errors = stats.clone();
    Ok(device.build_input_stream(
        config,
        move |data: &[T], _| {
            let Ok(gate) = accepting.try_lock() else {
                stats
                    .dropped
                    .fetch_add((data.len() / channels) as u64, Ordering::Relaxed);
                return;
            };
            if !*gate {
                return;
            }
            let mono = data
                .chunks_exact(channels)
                .map(|frame| {
                    frame
                        .iter()
                        .map(|value| <f32 as cpal::Sample>::from_sample(*value))
                        .sum::<f32>()
                        / channels as f32
                })
                .collect();
            enqueue(&tx, mono, &stats);
        },
        move |_| {
            errors.errors.fetch_add(1, Ordering::Relaxed);
        },
        None,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overflowing_queue_counts_exact_samples() {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let stats = CaptureStats::default();
        enqueue(&tx, vec![0.1; 5], &stats);
        enqueue(&tx, vec![0.2; 7], &stats);
        assert_eq!(stats.dropped.load(Ordering::Relaxed), 7);
        assert_eq!(rx.recv().unwrap().len(), 5);
    }
}
