use df::tract::{DfParams, DfTract, ReduceMask, RuntimeParams};
use ndarray::Array2;

use crate::engine::{Backend, BackendFactory};

pub const HOP: usize = 480;
pub const DELAY: usize = 480;
pub const SETTLE_FRAMES: usize = 1;

const CONTROL_DEFAULTS: [f32; 5] = [100.0, -15.0, 35.0, 35.0, 0.0];
const CONTROL_BOUNDS: [(f32, f32); 5] = [
    (0.0, 100.0),
    (-15.0, 35.0),
    (-15.0, 35.0),
    (-15.0, 35.0),
    (0.0, 0.05),
];

pub struct DfnBackend {
    live: DfTract,
    pristine: DfTract,
    input: Array2<f32>,
    output: Array2<f32>,
    controls: [f32; 5],
}

impl DfnBackend {
    pub fn new() -> Result<Self, String> {
        let params = RuntimeParams::new(1, 0.0, 100.0, -15.0, 35.0, 35.0, ReduceMask::MEAN);
        let live = DfTract::new(DfParams::default(), &params)
            .map_err(|error| format!("could not initialize DeepFilterNet3-LL: {error}"))?;
        let pristine = live.clone();
        Ok(Self {
            live,
            pristine,
            input: Array2::zeros((1, HOP)),
            output: Array2::zeros((1, HOP)),
            controls: CONTROL_DEFAULTS,
        })
    }

    pub fn factory() -> BackendFactory {
        Box::new(|| Ok(Box::new(Self::new()?) as Box<dyn Backend>))
    }

    pub(crate) fn attenuation_bypasses(value: f32) -> bool {
        value.is_finite() && value.clamp(0.0, 100.0) < 0.01
    }

    fn normalized_control(index: usize, value: f32) -> Option<f32> {
        let default = *CONTROL_DEFAULTS.get(index)?;
        let (min, max) = *CONTROL_BOUNDS.get(index)?;
        Some(if value.is_finite() {
            value.clamp(min, max)
        } else {
            default
        })
    }

    fn apply_control(&mut self, index: usize, value: f32) {
        let Some(value) = Self::normalized_control(index, value) else {
            return;
        };
        // Below 0.01 dB libDF passes input through with zero delay and frozen history; the
        // engine renders aligned dry instead, so keep processing at the last valid limit.
        if index == 0 && Self::attenuation_bypasses(value) {
            return;
        }
        if self.controls[index].to_bits() == value.to_bits() {
            return;
        }
        self.controls[index] = value;
        match index {
            0 => self.live.set_atten_lim(value),
            1 => self.live.min_db_thresh = value,
            2 => self.live.max_db_erb_thresh = value,
            3 => self.live.max_db_df_thresh = value,
            4 => self.live.set_pf_beta(value),
            _ => {}
        }
    }
}

impl Backend for DfnBackend {
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
        self.input
            .row_mut(0)
            .as_slice_mut()
            .ok_or_else(|| "non-contiguous DeepFilterNet input frame".to_owned())?
            .copy_from_slice(input);
        self.live
            .process(self.input.view(), self.output.view_mut())
            .map_err(|error| format!("DeepFilterNet process failed: {error}"))?;
        output.copy_from_slice(
            self.output
                .as_slice()
                .ok_or_else(|| "non-contiguous DeepFilterNet output frame".to_owned())?,
        );
        Ok(())
    }

    fn reset(&mut self) {
        self.live = self.pristine.clone();
        let controls = self.controls;
        self.controls = CONTROL_DEFAULTS;
        for (index, value) in controls.into_iter().enumerate() {
            self.apply_control(index, value);
        }
    }

    fn set_control(&mut self, index: usize, value: f32) {
        self.apply_control(index, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::TEST_LOCK;

    fn process_frames(backend: &mut DfnBackend, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len() % HOP, 0);
        let mut output = vec![0.0; input.len()];
        let (sources, input_remainder) = input.as_chunks::<HOP>();
        let (targets, output_remainder) = output.as_chunks_mut::<HOP>();
        assert!(input_remainder.is_empty() && output_remainder.is_empty());
        for (source, target) in sources.iter().zip(targets) {
            backend
                .process(source, target)
                .expect("DeepFilterNet frame");
        }
        output
    }

    fn broadband(frames: usize) -> Vec<f32> {
        let mut state = 0x8a5c_13d7_9b4e_206f_u64;
        (0..frames * HOP)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / ((1_u32 << 24) - 1) as f32;
                (unit - 0.5) * 0.4
            })
            .collect()
    }

    #[test]
    fn low_attenuation_broadband_correlation_measures_480_sample_delay() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut backend = DfnBackend::new().expect("DFN3-LL backend");
        backend.set_control(0, 1.0);
        backend.reset();
        let input = broadband(40);
        let output = process_frames(&mut backend, &input);

        // At 1 dB attenuation libDF mixes at least 10^(-1/20) of its aligned
        // noisy path into the result, so broadband correlation exposes STFT delay
        // even when the model rejects an impulse as noise.
        let start = 3 * HOP;
        let mut best = (f64::NEG_INFINITY, 0usize);
        for lag in 0..=2 * HOP {
            let mut dot = 0.0f64;
            let mut input_power = 0.0f64;
            let mut output_power = 0.0f64;
            for index in start..input.len() {
                let x = input[index - lag] as f64;
                let y = output[index] as f64;
                dot += x * y;
                input_power += x * x;
                output_power += y * y;
            }
            let correlation = dot / (input_power * output_power).sqrt();
            if correlation > best.0 {
                best = (correlation, lag);
            }
        }
        assert_eq!(best.1, DELAY, "best normalized correlation: {}", best.0);
        assert!(best.0 > 0.8, "correlation too weak: {}", best.0);
        assert!(output.iter().all(|sample| sample.is_finite()));
    }

    fn correlation(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let power = |s: &[f32]| s.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
        dot / (power(a) * power(b)).sqrt().max(f64::MIN_POSITIVE)
    }

    #[test]
    fn first_valid_frame_after_reset_matches_settle_frames() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut backend = DfnBackend::new().expect("DFN3-LL backend");
        backend.set_control(0, 1.0);
        // Post-gap stream: frame 0 is the audio the backend never saw (dropped before reset).
        let stream = broadband(9);
        let _ = process_frames(&mut backend, &broadband(4));
        backend.reset();
        let output = process_frames(&mut backend, &stream[HOP..]);
        // Output frame j aligns with stream frame j (D = HOP); valid once it tracks it.
        let first_valid = (0..8)
            .find(|&j| {
                correlation(
                    &output[j * HOP..(j + 1) * HOP],
                    &stream[j * HOP..(j + 1) * HOP],
                ) > 0.8
            })
            .expect("a valid frame");
        assert_eq!(first_valid, backend.settle_frames());
    }

    #[test]
    fn zero_attenuation_never_reaches_libdf_passthrough() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut backend = DfnBackend::new().expect("DFN3-LL backend");
        backend.set_control(0, 12.0);
        let limit = backend.live.atten_lim;
        backend.set_control(0, 0.0);
        backend.reset();
        assert_eq!(backend.live.atten_lim, limit);
        assert_ne!(backend.live.atten_lim, Some(1.0));
    }

    #[test]
    fn reset_replays_identically_after_prior_audio_and_gap_silence() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut backend = DfnBackend::new().expect("DFN3-LL backend");
        backend.set_control(0, 1.0);
        backend.set_control(4, 0.02);
        let old_audio = broadband(8);
        let silence = vec![0.0; HOP];
        let signal = broadband(8);

        backend.reset();
        let _ = process_frames(&mut backend, &old_audio);
        // Model the worker's gap path: full reset, one settling silence frame,
        // then the same signal. Replaying that sequence must be bit-identical.
        backend.reset();
        assert!((backend.live.atten_lim.unwrap() - 10.0f32.powf(-1.0 / 20.0)).abs() < 1e-6);
        assert_eq!(backend.live.post_filter_beta, 0.02);
        let first_silence = process_frames(&mut backend, &silence);
        let first_signal = process_frames(&mut backend, &signal);
        backend.reset();
        let replay_silence = process_frames(&mut backend, &silence);
        let replay_signal = process_frames(&mut backend, &signal);

        assert_eq!(first_silence, replay_silence);
        assert_eq!(first_signal, replay_signal);
        assert!(first_signal.iter().all(|sample| sample.is_finite()));
    }
}
