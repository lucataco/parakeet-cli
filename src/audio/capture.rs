/// Audio capture from microphone via cpal.
///
/// Provides a streaming audio source that captures from the system's
/// default or named input device, converts to mono f32 at the device's
/// native sample rate, and sends chunks through a crossbeam channel.
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender};

/// Audio chunk sent from the capture callback to the consumer.
pub struct AudioChunk {
    /// Mono f32 samples at the capture sample rate.
    pub samples: Vec<f32>,
}

/// Handle to an active audio capture stream.
pub struct CaptureStream {
    /// The underlying cpal stream (keeps it alive).
    _stream: cpal::Stream,
    /// Receiver for audio chunks.
    pub receiver: Receiver<AudioChunk>,
    /// The sample rate of the captured audio.
    pub sample_rate: u32,
}

/// Print a formatted list of audio input devices.
pub fn print_input_devices() -> Result<()> {
    let host = cpal::default_host();

    let default_device = host.default_input_device();
    let default_name = default_device
        .as_ref()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();

    println!("Audio input devices:");
    println!();

    for device in host
        .input_devices()
        .map_err(|e| anyhow::anyhow!("Failed to enumerate input devices: {e}"))?
    {
        let name = device.name().unwrap_or_else(|_| "<unknown>".to_string());

        let is_default = name == default_name;
        let marker = if is_default { " (default)" } else { "" };

        if let Ok(config) = device.default_input_config() {
            println!(
                "  {}{}: {}ch, {}Hz, {:?}",
                name,
                marker,
                config.channels(),
                config.sample_rate().0,
                config.sample_format()
            );
        } else {
            println!("  {}{}: <no supported config>", name, marker);
        }
    }

    Ok(())
}

/// Find an input device by name, or return the default.
fn find_input_device(device_name: &Option<String>) -> Result<cpal::Device> {
    let host = cpal::default_host();

    match device_name {
        Some(name) => {
            let devices = host
                .input_devices()
                .map_err(|e| anyhow::anyhow!("Failed to enumerate input devices: {e}"))?;

            for device in devices {
                if let Ok(dev_name) = device.name() {
                    if dev_name.contains(name.as_str()) {
                        return Ok(device);
                    }
                }
            }
            anyhow::bail!(
                "Input device '{}' not found. Run `parakeet devices` to list available devices.",
                name
            );
        }
        None => host
            .default_input_device()
            .context("No default input device available"),
    }
}

/// Start capturing audio from the specified (or default) input device.
///
/// Returns a CaptureStream handle. Audio flows through the channel as
/// AudioChunk messages. The stream runs until the CaptureStream is dropped.
pub fn start_capture(device_name: &Option<String>) -> Result<CaptureStream> {
    let device = find_input_device(device_name)?;
    let dev_name = device.name().unwrap_or_else(|_| "<unknown>".to_string());

    let config = device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("Failed to get input config for '{}': {e}", dev_name))?;

    let sample_rate = config.sample_rate().0;
    let channels = config.channels();
    let sample_format = config.sample_format();

    eprintln!(
        "Capturing from: {} ({}ch, {}Hz, {:?})",
        dev_name, channels, sample_rate, sample_format
    );

    // Channel for sending audio chunks from callback to consumer.
    // Bounded to ~2 seconds of buffering at typical chunk sizes to avoid unbounded growth.
    let (tx, rx): (Sender<AudioChunk>, Receiver<AudioChunk>) = crossbeam_channel::bounded(200);

    let ch = channels;
    let stream_config = config.into();

    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_input_stream(&device, &stream_config, ch, tx, |s: f32| s)?,
        cpal::SampleFormat::F64 => {
            build_input_stream(&device, &stream_config, ch, tx, |s: f64| s as f32)?
        }
        cpal::SampleFormat::I8 => {
            build_input_stream(&device, &stream_config, ch, tx, |s: i8| s as f32 / 128.0)?
        }
        cpal::SampleFormat::I16 => {
            build_input_stream(&device, &stream_config, ch, tx, |s: i16| s as f32 / 32768.0)?
        }
        cpal::SampleFormat::I32 => {
            build_input_stream(&device, &stream_config, ch, tx, |s: i32| {
                s as f32 / 2_147_483_648.0
            })?
        }
        cpal::SampleFormat::I64 => {
            build_input_stream(&device, &stream_config, ch, tx, |s: i64| {
                (s as f64 / 9_223_372_036_854_775_808.0) as f32
            })?
        }
        cpal::SampleFormat::U8 => build_input_stream(&device, &stream_config, ch, tx, |s: u8| {
            (s as f32 - 128.0) / 128.0
        })?,
        cpal::SampleFormat::U16 => {
            build_input_stream(&device, &stream_config, ch, tx, |s: u16| {
                (s as f32 - 32768.0) / 32768.0
            })?
        }
        cpal::SampleFormat::U32 => {
            build_input_stream(&device, &stream_config, ch, tx, |s: u32| {
                ((s as f64 - 2_147_483_648.0) / 2_147_483_648.0) as f32
            })?
        }
        cpal::SampleFormat::U64 => {
            build_input_stream(&device, &stream_config, ch, tx, |s: u64| {
                ((s as f64 - 9_223_372_036_854_775_808.0) / 9_223_372_036_854_775_808.0) as f32
            })?
        }
        fmt => anyhow::bail!("Unsupported sample format: {:?}", fmt),
    };

    stream
        .play()
        .map_err(|e| anyhow::anyhow!("Failed to start audio stream: {e}"))?;

    Ok(CaptureStream {
        _stream: stream,
        receiver: rx,
        sample_rate,
    })
}

fn build_input_stream<T, F>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: u16,
    tx: Sender<AudioChunk>,
    convert: F,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + Copy,
    F: Fn(T) -> f32 + Send + 'static,
{
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let samples: Vec<f32> = data.iter().map(|&sample| convert(sample)).collect();
                let mono = to_mono_f32(&samples, channels);
                let _ = tx.try_send(AudioChunk { samples: mono });
            },
            |err| eprintln!("Audio capture error: {err}"),
            None,
        )
        .map_err(|e| anyhow::anyhow!("Failed to build input stream: {e}"))
}

/// Convert interleaved multi-channel audio to mono by averaging.
fn to_mono_f32(data: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }
    let ch = channels as usize;
    let n_frames = data.len() / ch;
    let mut mono = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let mut sum = 0.0f32;
        for c in 0..ch {
            sum += data[i * ch + c];
        }
        mono.push(sum / ch as f32);
    }
    mono
}
