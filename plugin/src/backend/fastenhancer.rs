use std::path::PathBuf;
use std::sync::OnceLock;

use ort::ep::CPU;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{Tensor, TensorElementType};

use crate::engine::{Backend, BackendFactory};

pub const HOP: usize = 512;
pub const DELAY: usize = 512;
pub const SETTLE_FRAMES: usize = 1;

const CACHE_OUTPUTS: [&str; 5] = [
    "cache_out_0",
    "cache_out_1",
    "cache_out_2",
    "cache_out_3",
    "cache_out_4",
];

#[derive(Clone, Copy)]
enum Model {
    B,
    S,
}

impl Model {
    fn name(self) -> &'static str {
        match self {
            Self::B => "FastEnhancer-B",
            Self::S => "FastEnhancer-S",
        }
    }

    fn bytes(self) -> &'static [u8] {
        match self {
            Self::B => include_bytes!("../../models/fastenhancer_b.onnx"),
            Self::S => include_bytes!("../../models/fastenhancer_s.onnx"),
        }
    }

    fn cache_width(self) -> usize {
        match self {
            Self::B => 36,
            Self::S => 48,
        }
    }
}

pub struct FastEnhancer {
    session: Session,
    inputs: [Tensor<f32>; 6],
}

impl FastEnhancer {
    pub fn factory_b() -> BackendFactory {
        Self::factory(Model::B)
    }

    pub fn factory_s() -> BackendFactory {
        Self::factory(Model::S)
    }

    fn factory(model: Model) -> BackendFactory {
        Box::new(move || Ok(Box::new(Self::new(model)?) as Box<dyn Backend>))
    }

    pub fn new_b() -> Result<Self, String> {
        Self::new(Model::B)
    }

    pub fn new_s() -> Result<Self, String> {
        Self::new(Model::S)
    }

    fn new(model: Model) -> Result<Self, String> {
        initialize_ort()?;
        let mut builder = Session::builder()
            .map_err(|error| format!("could not create {} session: {error}", model.name()))?;
        builder = builder
            .with_no_environment_execution_providers()
            .map_err(|error| error.to_string())?
            .with_execution_providers([CPU::default().build()])
            .map_err(|error| error.to_string())?
            .with_intra_threads(1)
            .map_err(|error| error.to_string())?
            .with_inter_threads(1)
            .map_err(|error| error.to_string())?
            .with_parallel_execution(false)
            .map_err(|error| error.to_string())?
            .with_intra_op_spinning(false)
            .map_err(|error| error.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| error.to_string())?;
        let session = builder
            .commit_from_memory(model.bytes())
            .map_err(|error| format!("could not load {} model: {error}", model.name()))?;
        validate_io(&session, model)?;

        let cache_width = model.cache_width();
        let cache_size = cache_width * cache_width;
        let inputs = [
            tensor_2d(HOP)?,
            tensor_2d(HOP)?,
            tensor_2d(HOP)?,
            tensor_3d(cache_width, cache_size)?,
            tensor_3d(cache_width, cache_size)?,
            tensor_3d(cache_width, cache_size)?,
        ];
        Ok(Self { session, inputs })
    }
}

fn initialize_ort() -> Result<(), String> {
    static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();
    ORT_INIT
        .get_or_init(|| {
            let library = std::env::var_os("CLEARVOICE_ORT_DYLIB")
                .map(PathBuf::from)
                .unwrap_or_else(|| "/usr/lib/libonnxruntime.so.1".into());
            let environment = ort::init_from(&library)
                .map_err(|error| format!("could not load {}: {error}", library.display()))?;
            if !environment.commit() {
                return Err("ONNX Runtime environment was already configured".to_owned());
            }
            Ok(())
        })
        .clone()
}

fn tensor_2d(width: usize) -> Result<Tensor<f32>, String> {
    Tensor::from_array(([1usize, width], vec![0.0f32; width]))
        .map_err(|error| format!("could not allocate FastEnhancer input tensor: {error}"))
}

fn tensor_3d(width: usize, size: usize) -> Result<Tensor<f32>, String> {
    Tensor::from_array(([1usize, width, width], vec![0.0f32; size]))
        .map_err(|error| format!("could not allocate FastEnhancer cache tensor: {error}"))
}

fn validate_io(session: &Session, model: Model) -> Result<(), String> {
    let width = model.cache_width() as i64;
    let inputs = [
        ("wav_in", &[1, HOP as i64][..]),
        ("cache_in_0", &[1, HOP as i64][..]),
        ("cache_in_1", &[1, HOP as i64][..]),
        ("cache_in_2", &[1, width, width][..]),
        ("cache_in_3", &[1, width, width][..]),
        ("cache_in_4", &[1, width, width][..]),
    ];
    let outputs = [
        ("wav_out", &[1, HOP as i64][..]),
        ("cache_out_0", &[1, HOP as i64][..]),
        ("cache_out_1", &[1, HOP as i64][..]),
        ("cache_out_2", &[1, width, width][..]),
        ("cache_out_3", &[1, width, width][..]),
        ("cache_out_4", &[1, width, width][..]),
    ];
    for (direction, actual, expected) in [
        ("input", session.inputs(), &inputs),
        ("output", session.outputs(), &outputs),
    ] {
        if actual.len() != expected.len() {
            return Err(format!(
                "{} has {} {direction}s, expected {}",
                model.name(),
                actual.len(),
                expected.len()
            ));
        }
        for (index, (outlet, (name, shape))) in actual.iter().zip(expected).enumerate() {
            let actual_shape = outlet.dtype().tensor_shape().map(|shape| &**shape);
            if outlet.name() != *name
                || outlet.dtype().tensor_type() != Some(TensorElementType::Float32)
                || actual_shape != Some(*shape)
            {
                return Err(format!(
                    "{} {direction} {index} mismatch: name={} type={:?} shape={actual_shape:?}",
                    model.name(),
                    outlet.name(),
                    outlet.dtype().tensor_type()
                ));
            }
        }
    }
    Ok(())
}

impl Backend for FastEnhancer {
    fn hop(&self) -> usize {
        HOP
    }

    fn delay(&self) -> usize {
        DELAY
    }

    fn settle_frames(&self) -> usize {
        SETTLE_FRAMES
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        if input.len() != HOP || output.len() != HOP {
            return Err(format!(
                "expected {HOP}-sample frame, got input={} output={}",
                input.len(),
                output.len()
            ));
        }
        self.inputs[0].extract_tensor_mut().1.copy_from_slice(input);
        let outputs = self
            .session
            .run(ort::inputs![
                self.inputs[0].upcast_ref(),
                self.inputs[1].upcast_ref(),
                self.inputs[2].upcast_ref(),
                self.inputs[3].upcast_ref(),
                self.inputs[4].upcast_ref(),
                self.inputs[5].upcast_ref(),
            ])
            .map_err(|error| format!("FastEnhancer inference failed: {error}"))?;
        let waveform = outputs
            .get("wav_out")
            .ok_or_else(|| "FastEnhancer output is missing wav_out".to_owned())?
            .try_extract_tensor::<f32>()
            .map_err(|error| format!("invalid FastEnhancer wav_out: {error}"))?
            .1;
        if waveform.len() != HOP || !waveform.iter().all(|sample| sample.is_finite()) {
            return Err("FastEnhancer produced invalid wav_out".to_owned());
        }
        output.copy_from_slice(waveform);
        for (index, name) in CACHE_OUTPUTS.iter().enumerate() {
            let values = outputs
                .get(*name)
                .ok_or_else(|| format!("FastEnhancer output is missing {name}"))?
                .try_extract_tensor::<f32>()
                .map_err(|error| format!("invalid FastEnhancer {name}: {error}"))?
                .1;
            let cache = self.inputs[index + 1].extract_tensor_mut().1;
            if values.len() != cache.len() {
                return Err(format!("FastEnhancer produced invalid {name} size"));
            }
            cache.copy_from_slice(values);
        }
        Ok(())
    }

    fn reset(&mut self) {
        for cache in &mut self.inputs[1..] {
            cache.extract_tensor_mut().1.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TEST_LOCK;

    type BackendConstructor = fn() -> Result<FastEnhancer, String>;

    fn models() -> [(Model, BackendConstructor); 2] {
        [
            (Model::B, FastEnhancer::new_b),
            (Model::S, FastEnhancer::new_s),
        ]
    }

    fn process_frames(backend: &mut FastEnhancer, input: &[f32]) -> Vec<f32> {
        assert!(input.len().is_multiple_of(HOP));
        let mut output = vec![0.0; input.len()];
        let (sources, input_remainder) = input.as_chunks::<HOP>();
        let (targets, output_remainder) = output.as_chunks_mut::<HOP>();
        assert!(input_remainder.is_empty() && output_remainder.is_empty());
        for (source, target) in sources.iter().zip(targets) {
            backend.process(source, target).expect("FastEnhancer frame");
        }
        output
    }

    fn speech_signal(frames: usize) -> Vec<f32> {
        let mut state = 0x8a5c_13d7_9b4e_206f_u64;
        (0..frames * HOP)
            .map(|sample| {
                let time = sample as f64 / 48_000.0;
                let phase = std::f64::consts::TAU * 140.0 * time;
                let voice = (1..=11)
                    .map(|harmonic| (phase * harmonic as f64).sin() / harmonic as f64)
                    .sum::<f64>();
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let noise = (state >> 40) as f64 / ((1_u32 << 24) - 1) as f64 - 0.5;
                let envelope = 0.5 + 0.5 * (std::f64::consts::TAU * 3.0 * time).sin().max(0.0);
                (voice * envelope * 0.06 + noise * 0.06) as f32
            })
            .collect()
    }

    fn normalized_correlation(left: &[f32], right: &[f32]) -> f64 {
        let dot: f64 = left
            .iter()
            .zip(right)
            .map(|(x, y)| *x as f64 * *y as f64)
            .sum();
        let power = |samples: &[f32]| {
            samples
                .iter()
                .map(|sample| (*sample as f64).powi(2))
                .sum::<f64>()
        };
        dot / (power(left) * power(right)).sqrt().max(f64::MIN_POSITIVE)
    }

    fn measured_delay(input: &[f32], output: &[f32]) -> (usize, f64) {
        let start = 2 * 48_000;
        (0..=2 * HOP)
            .map(|lag| {
                let count = input.len().saturating_sub(start + lag);
                let correlation = normalized_correlation(
                    &input[start..start + count],
                    &output[start + lag..start + lag + count],
                );
                (lag, correlation)
            })
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("delay candidates")
    }

    #[test]
    fn speech_correlation_measures_delay_and_finite_stream_for_both_models() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let input = speech_signal(8 * 48_000 / HOP);
        for (model, new_backend) in models() {
            let mut backend = new_backend().expect("FastEnhancer session");
            let output = process_frames(&mut backend, &input);
            assert!(output.iter().all(|sample| sample.is_finite()));
            let (delay, correlation) = measured_delay(&input, &output);
            assert_eq!(delay, DELAY, "{} correlation={correlation}", model.name());
            assert!(
                correlation > 0.1,
                "{} correlation={correlation}",
                model.name()
            );
        }
    }

    #[test]
    fn reset_replay_is_bit_identical_after_prior_audio_gap_and_silence() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = speech_signal(16);
        let mut after_gap = vec![0.0; 3 * HOP];
        after_gap.extend(speech_signal(16));
        for (model, new_backend) in models() {
            let mut backend = new_backend().expect("FastEnhancer session");
            let _ = process_frames(&mut backend, &previous);
            backend.reset();
            let first = process_frames(&mut backend, &after_gap);
            let _ = process_frames(&mut backend, &previous);
            backend.reset();
            let replay = process_frames(&mut backend, &after_gap);
            assert_eq!(first, replay, "{} reset replay", model.name());
            assert!(first.iter().all(|sample| sample.is_finite()));
        }
    }

    #[test]
    fn reset_matches_fresh_backend_after_different_audio_and_derives_settling_frame() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prior: Vec<f32> = speech_signal(8).into_iter().map(|sample| -sample).collect();
        let stream = speech_signal(13);
        let post_reset_input = &stream[HOP..];
        for (model, new_backend) in models() {
            let mut backend = new_backend().expect("FastEnhancer session");
            let mut fresh = new_backend().expect("fresh FastEnhancer session");
            let _ = process_frames(&mut backend, &prior);
            backend.reset();
            let output = process_frames(&mut backend, post_reset_input);
            let oracle = process_frames(&mut fresh, post_reset_input);
            assert_eq!(output, oracle, "{} reset vs fresh backend", model.name());
            assert!(output.iter().all(|sample| sample.is_finite()));

            let first_valid = (0..oracle.len() / HOP)
                .find(|frame| {
                    normalized_correlation(
                        &oracle[frame * HOP..(frame + 1) * HOP],
                        &stream[frame * HOP..(frame + 1) * HOP],
                    ) > 0.2
                })
                .expect("a valid post-reset frame");
            assert_eq!(
                first_valid,
                SETTLE_FRAMES,
                "{} first valid frame",
                model.name()
            );
        }
    }
}
