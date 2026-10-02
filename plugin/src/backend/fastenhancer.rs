use ort::session::{Session, SessionInputValue};
use ort::value::{Outlet, Tensor, TensorElementType};

use crate::backend::build_ort_session;
use crate::engine::{Backend, BackendFactory};

// ponytail: six-cache ceiling matches shipped B/S/M; raise it with a new model.
const MAX_CACHES: usize = 6;
const MAX_INPUTS: usize = MAX_CACHES + 1;

#[derive(Clone, Copy)]
enum Model {
    B,
    S,
    M,
}

impl Model {
    fn name(self) -> &'static str {
        match self {
            Self::B => "FastEnhancer-B",
            Self::S => "FastEnhancer-S",
            Self::M => "FastEnhancer-M",
        }
    }

    fn bytes(self) -> &'static [u8] {
        match self {
            Self::B => include_bytes!("../../models/fastenhancer_b.onnx"),
            Self::S => include_bytes!("../../models/fastenhancer_s.onnx"),
            Self::M => include_bytes!("../../models/fastenhancer_m.onnx"),
        }
    }
}

struct ModelIo {
    hop: usize,
    delay: usize,
    input_shapes: Vec<Vec<usize>>,
    cache_output_names: Vec<String>,
}

pub struct FastEnhancer {
    session: Session,
    inputs: Vec<Tensor<f32>>,
    cache_output_names: Vec<String>,
    hop: usize,
    delay: usize,
}

impl FastEnhancer {
    pub fn factory_b() -> BackendFactory {
        Self::factory(Model::B)
    }

    pub fn factory_s() -> BackendFactory {
        Self::factory(Model::S)
    }

    pub fn factory_m() -> BackendFactory {
        Self::factory(Model::M)
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

    pub fn new_m() -> Result<Self, String> {
        Self::new(Model::M)
    }

    fn new(model: Model) -> Result<Self, String> {
        let session = build_ort_session(model.name(), model.bytes())?;
        let io = validate_io(&session, model)?;
        let inputs = io
            .input_shapes
            .iter()
            .map(|shape| tensor(shape, model))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            session,
            inputs,
            cache_output_names: io.cache_output_names,
            hop: io.hop,
            delay: io.delay,
        })
    }
}

fn tensor(shape: &[usize], model: Model) -> Result<Tensor<f32>, String> {
    let size = shape
        .iter()
        .try_fold(1usize, |size, dimension| size.checked_mul(*dimension))
        .ok_or_else(|| format!("{} tensor size overflows usize", model.name()))?;
    Tensor::from_array((shape.to_vec(), vec![0.0f32; size]))
        .map_err(|error| format!("could not allocate {} input tensor: {error}", model.name()))
}

fn static_f32_shape(outlet: &Outlet, model: Model, direction: &str) -> Result<Vec<usize>, String> {
    if outlet.dtype().tensor_type() != Some(TensorElementType::Float32) {
        return Err(format!(
            "{} {direction} {} must be a float32 tensor",
            model.name(),
            outlet.name()
        ));
    }
    let shape = outlet.dtype().tensor_shape().ok_or_else(|| {
        format!(
            "{} {direction} {} must have a static tensor shape",
            model.name(),
            outlet.name()
        )
    })?;
    if shape.is_empty() {
        return Err(format!(
            "{} {direction} {} has an empty tensor shape",
            model.name(),
            outlet.name()
        ));
    }
    shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension)
                .ok()
                .filter(|dimension| *dimension > 0)
                .ok_or_else(|| {
                    format!(
                        "{} {direction} {} has non-static or invalid shape {shape:?}",
                        model.name(),
                        outlet.name()
                    )
                })
        })
        .collect()
}

fn validate_io(session: &Session, model: Model) -> Result<ModelIo, String> {
    let inputs = session.inputs();
    let outputs = session.outputs();
    if inputs.len() < 2 || outputs.len() != inputs.len() {
        return Err(format!(
            "{} has {} inputs and {} outputs; expected wav plus matching cache pairs",
            model.name(),
            inputs.len(),
            outputs.len()
        ));
    }
    let cache_count = inputs.len() - 1;
    if cache_count > MAX_CACHES {
        return Err(format!(
            "{} has {cache_count} caches; at most {MAX_CACHES} are supported",
            model.name()
        ));
    }
    if inputs[0].name() != "wav_in" {
        return Err(format!(
            "{} input 0 is {}, expected wav_in",
            model.name(),
            inputs[0].name()
        ));
    }
    if outputs[0].name() != "wav_out" {
        return Err(format!(
            "{} output 0 is {}, expected wav_out",
            model.name(),
            outputs[0].name()
        ));
    }
    let wav_shape = static_f32_shape(&inputs[0], model, "input")?;
    let hop = match wav_shape.as_slice() {
        [1, hop] if *hop <= 512 => *hop,
        _ => {
            return Err(format!(
                "{} wav_in must have static shape [1, hop] with 1 <= hop <= 512, got {wav_shape:?}",
                model.name()
            ));
        }
    };
    if static_f32_shape(&outputs[0], model, "output")? != wav_shape {
        return Err(format!(
            "{} wav_out shape does not match wav_in shape {wav_shape:?}",
            model.name()
        ));
    }

    let mut input_shapes = vec![wav_shape];
    let mut cache_output_names = Vec::with_capacity(cache_count);
    let mut delay = None;
    for index in 0..cache_count {
        let input = &inputs[index + 1];
        let output = &outputs[index + 1];
        let expected_input = format!("cache_in_{index}");
        let expected_output = format!("cache_out_{index}");
        if input.name() != expected_input {
            return Err(format!(
                "{} cache input {index} is {}, expected {expected_input}",
                model.name(),
                input.name()
            ));
        }
        if output.name() != expected_output {
            return Err(format!(
                "{} cache output {index} is {}, expected {expected_output}",
                model.name(),
                output.name()
            ));
        }
        let input_shape = static_f32_shape(input, model, "input")?;
        let output_shape = static_f32_shape(output, model, "output")?;
        if input_shape[0] != 1 || input_shape != output_shape {
            return Err(format!(
                "{} {expected_input}/{expected_output} shapes must match and have batch 1, got {input_shape:?} and {output_shape:?}",
                model.name()
            ));
        }
        if index == 0 {
            delay = match input_shape.as_slice() {
                [1, width] => Some(*width),
                _ => {
                    return Err(format!(
                        "{} cache_in_0 must have static shape [1, delay], got {input_shape:?}",
                        model.name()
                    ));
                }
            };
        }
        input_shapes.push(input_shape);
        cache_output_names.push(output.name().to_owned());
    }

    let delay = delay.ok_or_else(|| format!("{} is missing cache_in_0", model.name()))?;
    Ok(ModelIo {
        hop,
        delay,
        input_shapes,
        cache_output_names,
    })
}

impl Backend for FastEnhancer {
    fn hop(&self) -> usize {
        self.hop
    }

    fn delay(&self) -> usize {
        self.delay
    }

    /// Output frame j after a reset covers input [j*hop - delay, (j+1)*hop - delay); frames
    /// reaching back before the reset (zero history) must render dry: B/S 1, M 3.
    fn settle_frames(&self) -> usize {
        self.delay.div_ceil(self.hop)
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        if input.len() != self.hop || output.len() != self.hop {
            return Err(format!(
                "expected {}-sample frame, got input={} output={}",
                self.hop,
                input.len(),
                output.len()
            ));
        }
        self.inputs[0].extract_tensor_mut().1.copy_from_slice(input);
        let input_count = self.inputs.len();
        let input_values: [SessionInputValue<'_>; MAX_INPUTS] = std::array::from_fn(|index| {
            self.inputs[index.min(input_count - 1)].upcast_ref().into()
        });
        let outputs = self
            .session
            .run(&input_values[..input_count])
            .map_err(|error| format!("FastEnhancer inference failed: {error}"))?;
        let waveform = outputs
            .get("wav_out")
            .ok_or_else(|| "FastEnhancer output is missing wav_out".to_owned())?
            .try_extract_tensor::<f32>()
            .map_err(|error| format!("invalid FastEnhancer wav_out: {error}"))?
            .1;
        if waveform.len() != self.hop || !waveform.iter().all(|sample| sample.is_finite()) {
            return Err("FastEnhancer produced invalid wav_out".to_owned());
        }
        output.copy_from_slice(waveform);
        for (index, name) in self.cache_output_names.iter().enumerate() {
            let values = outputs
                .get(name)
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

    fn models() -> [(Model, BackendConstructor, usize, usize); 3] {
        [
            (Model::B, FastEnhancer::new_b, 512, 512),
            (Model::S, FastEnhancer::new_s, 512, 512),
            (Model::M, FastEnhancer::new_m, 320, 704),
        ]
    }

    fn process_frames(backend: &mut FastEnhancer, input: &[f32]) -> Vec<f32> {
        let hop = backend.hop();
        assert!(input.len().is_multiple_of(hop));
        let mut output = vec![0.0; input.len()];
        for (source, target) in input.chunks_exact(hop).zip(output.chunks_exact_mut(hop)) {
            backend.process(source, target).expect("FastEnhancer frame");
        }
        output
    }

    fn speech_signal(samples: usize) -> Vec<f32> {
        let mut state = 0x8a5c_13d7_9b4e_206f_u64;
        (0..samples)
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

    fn measured_delay(input: &[f32], output: &[f32], max_lag: usize) -> (usize, f64) {
        let start = 2 * 48_000;
        (0..=max_lag)
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
    fn speech_correlation_measures_delay_and_finite_stream_for_all_models() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let input = speech_signal(8 * 48_000);
        for (model, new_backend, expected_hop, expected_delay) in models() {
            let mut backend = new_backend().expect("FastEnhancer session");
            assert_eq!(backend.hop(), expected_hop, "{} hop", model.name());
            assert_eq!(
                backend.delay(),
                expected_delay,
                "{} metadata delay",
                model.name()
            );
            let output = process_frames(&mut backend, &input);
            assert!(output.iter().all(|sample| sample.is_finite()));
            let (delay, correlation) = measured_delay(&input, &output, 2 * backend.delay());
            assert_eq!(
                delay,
                expected_delay,
                "{} correlation={correlation}",
                model.name()
            );
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
        for (model, new_backend, _, _) in models() {
            let mut backend = new_backend().expect("FastEnhancer session");
            let hop = backend.hop();
            let previous = speech_signal(16 * hop);
            let mut after_gap = vec![0.0; 3 * hop];
            after_gap.extend(speech_signal(16 * hop));
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
        for (model, new_backend, _, _) in models() {
            let mut backend = new_backend().expect("FastEnhancer session");
            let mut fresh = new_backend().expect("fresh FastEnhancer session");
            let hop = backend.hop();
            let settle_frames = backend.settle_frames();
            let prior: Vec<f32> = speech_signal(8 * hop)
                .into_iter()
                .map(|sample| -sample)
                .collect();
            let stream = speech_signal(13 * hop);
            let post_reset_input = &stream[hop..];
            let _ = process_frames(&mut backend, &prior);
            backend.reset();
            let output = process_frames(&mut backend, post_reset_input);
            let oracle = process_frames(&mut fresh, post_reset_input);
            assert_eq!(output, oracle, "{} reset vs fresh backend", model.name());
            assert!(output.iter().all(|sample| sample.is_finite()));

            // Output sample m aligns with post-reset input m - delay.
            let delay = backend.delay();
            assert_eq!(settle_frames, delay.div_ceil(hop), "{} G", model.name());
            let rms = |samples: &[f32]| {
                (samples.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / samples.len() as f64)
                    .sqrt()
            };
            // Frames from G on track the input they align with and carry real signal.
            let valid = &oracle[settle_frames * hop..(settle_frames + 4) * hop];
            for frame in settle_frames..settle_frames + 4 {
                let output = &oracle[frame * hop..(frame + 1) * hop];
                let aligned = &post_reset_input[frame * hop - delay..(frame + 1) * hop - delay];
                assert!(
                    normalized_correlation(output, aligned) > 0.2
                        && rms(output) > 0.1 * rms(aligned),
                    "{} frame {frame} should be valid",
                    model.name()
                );
            }
            // The output mapping to pre-reset time (zero history) is near-silent: rendering
            // any frame before G wet would be an audible dropout, so they must be dry.
            assert!(
                rms(&oracle[..delay]) < 0.1 * rms(valid),
                "{} pre-reset output rms {} vs valid {}",
                model.name(),
                rms(&oracle[..delay]),
                rms(valid)
            );
        }
    }
}
