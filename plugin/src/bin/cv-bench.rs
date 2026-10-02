use std::time::Instant;

use clearvoice_ladspa::Backend;
use clearvoice_ladspa::backend::dfn::{DfnBackend, HOP as DFN_HOP};
use clearvoice_ladspa::backend::fastenhancer::{
    DELAY as FE_DELAY, FastEnhancer, HOP as FE_HOP, SETTLE_FRAMES as FE_SETTLE_FRAMES,
};

const SAMPLE_RATE: usize = 48_000;
const DURATION_SECONDS: usize = 60;
const WARMUP_SECONDS: usize = 2;

fn main() {
    if let Err(error) = run() {
        eprintln!("cv-bench: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let backend = args.next();
    if args.next().is_some() {
        return Err("usage: cv-bench dfn3-ll | fastenhancer-b | fastenhancer-s".to_owned());
    }
    let backend_name = backend.as_deref().unwrap_or("");
    let construction_start = Instant::now();
    let (mut backend, hop, delay, settle_frames) = match backend_name {
        "dfn3-ll" => (
            Box::new(DfnBackend::new()?) as Box<dyn Backend>,
            DFN_HOP,
            clearvoice_ladspa::backend::dfn::DELAY,
            clearvoice_ladspa::backend::dfn::SETTLE_FRAMES,
        ),
        "fastenhancer-b" => (
            Box::new(FastEnhancer::new_b()?) as Box<dyn Backend>,
            FE_HOP,
            FE_DELAY,
            FE_SETTLE_FRAMES,
        ),
        "fastenhancer-s" => (
            Box::new(FastEnhancer::new_s()?) as Box<dyn Backend>,
            FE_HOP,
            FE_DELAY,
            FE_SETTLE_FRAMES,
        ),
        _ => return Err("usage: cv-bench dfn3-ll | fastenhancer-b | fastenhancer-s".to_owned()),
    };
    let mut warm_output = vec![0.0; hop];
    backend.process(&vec![0.0; hop], &mut warm_output)?;
    let reset_start = Instant::now();
    backend.reset();
    let reset_time = reset_start.elapsed();
    let ready_time = construction_start.elapsed();

    let signal = synthetic_signal(DURATION_SECONDS);
    if !signal.len().is_multiple_of(hop) {
        return Err(format!(
            "{backend_name} benchmark signal is not hop-aligned"
        ));
    }
    let warmup_frames = (WARMUP_SECONDS * SAMPLE_RATE).div_ceil(hop);
    let warmup_samples = warmup_frames * hop;
    let mut timings_ms = Vec::with_capacity((signal.len() - warmup_samples) / hop);
    let mut output = vec![0.0; hop];
    for frame in signal[..warmup_samples].chunks_exact(hop) {
        backend.process(frame, &mut output)?;
    }
    let measured_start = Instant::now();
    for frame in signal[warmup_samples..].chunks_exact(hop) {
        let started = Instant::now();
        backend.process(frame, &mut output)?;
        timings_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let measured_elapsed = measured_start.elapsed();

    timings_ms.sort_unstable_by(f64::total_cmp);
    let p50 = percentile(&timings_ms, 0.50);
    let p99 = percentile(&timings_ms, 0.99);
    let p999 = percentile(&timings_ms, 0.999);
    let max = timings_ms.last().copied().unwrap_or(0.0);
    let measured_audio_seconds = timings_ms.len() as f64 * hop as f64 / SAMPLE_RATE as f64;
    let rtf = measured_elapsed.as_secs_f64() / measured_audio_seconds;

    println!("backend={backend_name} hop={hop} delay={delay} settle_frames={settle_frames}");
    println!(
        "construct-to-ready (model + 1 silent warm frame + reset): {:.3} ms",
        ready_time.as_secs_f64() * 1000.0
    );
    if backend_name == "dfn3-ll" {
        println!(
            "reset snapshot clone: {:.3} ms",
            reset_time.as_secs_f64() * 1000.0
        );
        if reset_time > std::time::Duration::from_millis(5) {
            println!("WARNING: reset clone exceeds 5 ms");
        }
    } else {
        println!(
            "reset cache zero: {:.3} ms",
            reset_time.as_secs_f64() * 1000.0
        );
    }
    println!(
        "frame time over {} frames ({} s signal, {} s warm-up excluded): p50={p50:.4} ms p99={p99:.4} ms p99.9={p999:.4} ms max={max:.4} ms",
        timings_ms.len(),
        DURATION_SECONDS,
        WARMUP_SECONDS
    );
    println!("RTF={rtf:.4} (measured audio={measured_audio_seconds:.1} s)");
    Ok(())
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = (percentile * sorted.len() as f64).ceil() as usize;
    sorted[index.saturating_sub(1).min(sorted.len() - 1)]
}

struct NormalRng {
    state: u64,
    spare: Option<f64>,
}

impl NormalRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn uniform_open(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform_open().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform_open();
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

fn synthetic_signal(seconds: usize) -> Vec<f32> {
    let sample_count = seconds * SAMPLE_RATE;
    let gate_period = SAMPLE_RATE / 3;
    let mut rng = NormalRng::new(0);
    let mut signal = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        let phase = std::f64::consts::TAU * 140.0 * sample as f64 / SAMPLE_RATE as f64;
        let harmonics = (1..=11)
            .map(|harmonic| (phase * harmonic as f64).sin() / harmonic as f64)
            .sum::<f64>();
        let gate = if sample % gate_period < gate_period / 2 {
            1.0
        } else {
            0.0
        };
        signal.push((harmonics * gate * 0.3 * 0.2 + rng.normal() * 0.03) as f32);
    }
    signal
}

#[cfg(test)]
mod tests {
    use super::*;
    use clearvoice_ladspa::ladspa::{self, Descriptor, LadspaHandle};
    use std::sync::Mutex;
    use std::time::Duration;

    /// Real-time paced tests run one at a time: concurrent model workers (non-RT here) can
    /// push DFN3-LL past its deadline on a slow (battery) CPU.
    static PACED: Mutex<()> = Mutex::new(());

    struct PluginHandle {
        descriptor: *const Descriptor,
        handle: LadspaHandle,
    }

    impl Drop for PluginHandle {
        fn drop(&mut self) {
            unsafe {
                if let Some(deactivate) = (*self.descriptor).deactivate {
                    deactivate(self.handle);
                }
                if let Some(cleanup) = (*self.descriptor).cleanup {
                    cleanup(self.handle);
                }
            }
        }
    }

    #[test]
    fn descriptor_engine_runs_finite_wet_bench_signal_at_quantum_256() {
        let _paced = PACED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let signal = synthetic_signal(5);
        let descriptor_ptr = ladspa::ladspa_descriptor(0);
        assert!(!descriptor_ptr.is_null());
        let descriptor = unsafe { &*descriptor_ptr };
        let instantiate = descriptor.instantiate.expect("instantiate callback");
        let construct_start = Instant::now();
        let handle = unsafe { instantiate(descriptor_ptr, SAMPLE_RATE as _) };
        let ready_time = construct_start.elapsed();
        assert!(!handle.is_null(), "DFN3-LL instantiate failed");
        eprintln!(
            "DFN3-LL descriptor construct-to-ready: {:.3} ms",
            ready_time.as_secs_f64() * 1000.0
        );
        assert!(ready_time < Duration::from_secs(4));
        let _plugin = PluginHandle {
            descriptor: descriptor_ptr,
            handle,
        };

        let mut latency_ms = 35.0;
        let mut reported_latency = -1.0;
        let mut controls = [100.0, -15.0, 35.0, 35.0, 0.0];
        let mut input_block = [0.0; 256];
        let mut output_block = [0.0; 256];
        let mut rendered = vec![0.0; signal.len()];
        unsafe {
            let connect = descriptor.connect_port.expect("connect callback");
            connect(handle, 0, input_block.as_mut_ptr());
            connect(handle, 1, output_block.as_mut_ptr());
            connect(handle, 2, &mut latency_ms);
            connect(handle, 3, &mut reported_latency);
            for (index, value) in controls.iter_mut().enumerate() {
                connect(handle, (index + 4) as _, value);
            }
            descriptor.activate.expect("activate callback")(handle);
        }
        assert_eq!(reported_latency, 1680.0);

        let run = descriptor.run.expect("run callback");
        // Real-time pacing, like a PipeWire graph; faster feeding legitimately conceals.
        let started = Instant::now();
        for (block_index, samples) in signal.chunks(input_block.len()).enumerate() {
            let start = block_index * input_block.len();
            if start == 4 * SAMPLE_RATE {
                controls[0] = 0.0;
            }
            input_block[..samples.len()].copy_from_slice(samples);
            unsafe { run(handle, samples.len() as _) };
            rendered[start..start + samples.len()].copy_from_slice(&output_block[..samples.len()]);
            let due = Duration::from_secs_f64((start + samples.len()) as f64 / SAMPLE_RATE as f64);
            if let Some(wait) = due.checked_sub(started.elapsed()) {
                std::thread::sleep(wait);
            }
        }

        assert!(rendered.iter().all(|sample| sample.is_finite()));
        let mut residual_power = 0.0f64;
        let mut dry_power = 0.0f64;
        let wet_end = 4 * SAMPLE_RATE;
        for index in (2 * SAMPLE_RATE + reported_latency as usize)..wet_end {
            let dry = signal[index - reported_latency as usize] as f64;
            let residual = rendered[index] as f64 - dry;
            residual_power += residual * residual;
            dry_power += dry * dry;
        }
        let wet_ratio = residual_power / dry_power;
        eprintln!("DFN3-LL wet-to-dry residual power ratio after warm-up: {wet_ratio:.6}");
        assert!(wet_ratio > 1e-6, "engine output remained aligned dry");
        let bypass_start = 4 * SAMPLE_RATE + reported_latency as usize;
        assert!(
            rendered[bypass_start..]
                .iter()
                .zip(&signal[bypass_start - reported_latency as usize..])
                .all(|(actual, expected)| actual == expected)
        );
    }

    #[test]
    fn fastenhancer_descriptors_run_wet_at_realtime_quantum_256() {
        let _paced = PACED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let signal = synthetic_signal(2);
        for (index, model) in [(1, "FastEnhancer-B"), (2, "FastEnhancer-S")] {
            let descriptor_ptr = ladspa::ladspa_descriptor(index);
            assert!(!descriptor_ptr.is_null());
            let descriptor = unsafe { &*descriptor_ptr };
            let construct_start = Instant::now();
            let handle = unsafe {
                descriptor.instantiate.expect("instantiate callback")(
                    descriptor_ptr,
                    SAMPLE_RATE as _,
                )
            };
            let ready_time = construct_start.elapsed();
            assert!(!handle.is_null(), "{model} instantiate failed");
            eprintln!(
                "{model} descriptor construct-to-ready: {:.3} ms",
                ready_time.as_secs_f64() * 1000.0
            );
            assert!(ready_time < Duration::from_secs(4));
            let _plugin = PluginHandle {
                descriptor: descriptor_ptr,
                handle,
            };

            let mut reference_backend = match index {
                1 => FastEnhancer::new_b().expect("FastEnhancer-B reference backend"),
                2 => FastEnhancer::new_s().expect("FastEnhancer-S reference backend"),
                _ => unreachable!(),
            };
            reference_backend.reset();
            let mut reference = vec![0.0; signal.len()];
            let (source_frames, _) = signal.as_chunks::<FE_HOP>();
            let (reference_frames, _) = reference.as_chunks_mut::<FE_HOP>();
            for (source, target) in source_frames.iter().zip(reference_frames) {
                reference_backend
                    .process(source, target)
                    .expect("FastEnhancer reference frame");
            }

            let mut latency_ms = 35.0;
            let mut reported_latency = -1.0;
            let mut input_block = [0.0; 256];
            let mut output_block = [0.0; 256];
            let mut rendered = vec![0.0; signal.len()];
            unsafe {
                let connect = descriptor.connect_port.expect("connect callback");
                connect(handle, 0, input_block.as_mut_ptr());
                connect(handle, 1, output_block.as_mut_ptr());
                connect(handle, 2, &mut latency_ms);
                connect(handle, 3, &mut reported_latency);
                descriptor.activate.expect("activate callback")(handle);
            }
            assert_eq!(reported_latency, 1680.0);

            let run = descriptor.run.expect("run callback");
            let started = Instant::now();
            for (block_index, samples) in signal.chunks(input_block.len()).enumerate() {
                let start = block_index * input_block.len();
                input_block[..samples.len()].copy_from_slice(samples);
                unsafe { run(handle, samples.len() as _) };
                rendered[start..start + samples.len()]
                    .copy_from_slice(&output_block[..samples.len()]);
                let due =
                    Duration::from_secs_f64((start + samples.len()) as f64 / SAMPLE_RATE as f64);
                if let Some(wait) = due.checked_sub(started.elapsed()) {
                    std::thread::sleep(wait);
                }
            }

            assert!(rendered.iter().all(|sample| sample.is_finite()));
            let latency = reported_latency as usize;
            let alignment = latency - FE_DELAY;
            assert_eq!(alignment, 1_168);
            // Frame 1 is the first whose aligned dry input exists (frame 0 settles dry).
            let first_frame = 1;
            let mut compared_frames = 0;
            let mut reference_frames = 0;
            let mut output_energy = 0.0f64;
            let mut dry_energy = 0.0f64;
            let mut diverged = false;
            for frame in first_frame.. {
                if diverged {
                    break;
                }
                let output_start = alignment + frame * FE_HOP;
                let reference_start = frame * FE_HOP;
                if output_start + FE_HOP > rendered.len()
                    || reference_start + FE_HOP > reference.len()
                {
                    break;
                }
                let actual = &rendered[output_start..output_start + FE_HOP];
                let expected = &reference[reference_start..reference_start + FE_HOP];
                let dry_start = output_start - latency;
                let dry = &signal[dry_start..dry_start + FE_HOP];
                let matches_reference = actual
                    .iter()
                    .zip(expected)
                    .all(|(actual, expected)| (actual - expected).abs() <= 1e-6);
                if matches_reference {
                    reference_frames += 1;
                } else {
                    // Startup fade-in or a missed deadline: every sample must be a crossfade
                    // between the aligned reference and the aligned dry input. After a miss the
                    // worker may reset, so later wet frames follow a new history: stop there.
                    // ponytail: a miss before the first wet frame (startup under heavy load)
                    // still fails this test; deterministic alignment lives in engine tests.
                    for ((&actual, &wet), &dry) in actual.iter().zip(expected).zip(dry) {
                        assert!(
                            (actual - dry) * (wet - dry) >= -1e-9
                                && (actual - dry).abs() <= (wet - dry).abs() + 1e-6,
                            "{model} frame {frame}: {actual} is not a mix of wet {wet} and dry {dry}"
                        );
                    }
                    diverged = reference_frames > 0;
                }
                for (&actual, &dry) in actual.iter().zip(dry) {
                    output_energy += (actual as f64).powi(2);
                    dry_energy += (dry as f64).powi(2);
                }
                compared_frames += 1;
            }
            assert!(
                compared_frames > 0,
                "{model} compared no steady-state frames"
            );
            assert!(
                reference_frames > 0,
                "{model} had no reference-matched wet frames"
            );
            assert!(
                output_energy > dry_energy * 1e-6,
                "{model} engine output energy was negligible"
            );
        }
    }

    #[test]
    fn synthetic_signal_is_repeatable_and_finite() {
        let first = synthetic_signal(1);
        assert_eq!(first, synthetic_signal(1));
        assert_eq!(first.len(), SAMPLE_RATE);
        assert!(first.iter().all(|sample| sample.is_finite()));
    }
}
