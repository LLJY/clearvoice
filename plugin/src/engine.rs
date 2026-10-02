use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Once};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

use rtrb::{Consumer, Producer, RingBuffer};

const FRAME_CAPACITY: usize = 64;
const FRAME_SAMPLES: usize = 512;
const DRY_CAPACITY: usize = 16_384;
const FADE_SAMPLES: usize = 128;
const MAX_LATENCY: usize = 9_600;
const DEFAULT_LATENCY_MS: f32 = 35.0;
const THROTTLE_AFTER: Duration = Duration::from_millis(50);
const THROTTLE_SLEEP: Duration = Duration::from_millis(1);
const STATS_INTERVAL: Duration = Duration::from_secs(10);
/// Thread name while the backend is being built (never promoted to RT by ClearVoice).
const INIT_THREAD_NAME: &str = "cv-dsp-init";
/// Kernel thread name once the throttled worker loop runs (ClearVoice's RT target).
const WORKER_THREAD_NAME: &str = "cv-dsp-worker";
pub const BACKEND_CONTROL_SLOTS: usize = 8;

/// A stateful, worker-owned audio processor. Output sample m aligns with input
/// sample m - `delay()`; `reset()` must clear all history.
pub trait Backend: 'static {
    fn hop(&self) -> usize;
    fn delay(&self) -> usize;
    fn settle_frames(&self) -> usize;
    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String>;
    fn reset(&mut self);

    fn set_control(&mut self, _index: usize, _value: f32) {}
}

/// Factory closure is `Send`; the backend itself is created and stays on the worker.
pub type BackendFactory = Box<dyn FnOnce() -> Result<Box<dyn Backend>, String> + Send + 'static>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineStats {
    pub processed: u64,
    pub concealed: u64,
    pub discarded_late: u64,
    pub input_dropped: u64,
    pub output_dropped: u64,
    pub settling: u64,
    pub throttled: u64,
    pub oversized_blocks: u64,
}

#[derive(Default)]
struct Counters {
    processed: AtomicU64,
    concealed: AtomicU64,
    discarded_late: AtomicU64,
    input_dropped: AtomicU64,
    output_dropped: AtomicU64,
    settling: AtomicU64,
    throttled: AtomicU64,
    oversized_blocks: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> EngineStats {
        EngineStats {
            processed: self.processed.load(Ordering::Relaxed),
            concealed: self.concealed.load(Ordering::Relaxed),
            discarded_late: self.discarded_late.load(Ordering::Relaxed),
            input_dropped: self.input_dropped.load(Ordering::Relaxed),
            output_dropped: self.output_dropped.load(Ordering::Relaxed),
            settling: self.settling.load(Ordering::Relaxed),
            throttled: self.throttled.load(Ordering::Relaxed),
            oversized_blocks: self.oversized_blocks.load(Ordering::Relaxed),
        }
    }
}

struct Shared {
    epoch: AtomicU32,
    needed_frame: AtomicU64,
    stop: AtomicBool,
    dead: AtomicBool,
    bypass: AtomicBool,
    controls: [AtomicU32; BACKEND_CONTROL_SLOTS],
    counters: Counters,
}

impl Shared {
    fn new() -> Self {
        Self {
            epoch: AtomicU32::new(0),
            needed_frame: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            dead: AtomicBool::new(false),
            bypass: AtomicBool::new(false),
            controls: std::array::from_fn(|_| AtomicU32::new(0.0f32.to_bits())),
            counters: Counters::default(),
        }
    }
}

struct Frame {
    epoch: u32,
    index: u64,
    settling: bool,
    samples: [f32; FRAME_SAMPLES],
}

impl Frame {
    fn empty(epoch: u32, index: u64) -> Self {
        Self {
            epoch,
            index,
            settling: false,
            samples: [0.0; FRAME_SAMPLES],
        }
    }
}

struct Hooks {
    now: Arc<dyn Fn() -> Duration + Send + Sync>,
    sleep: Arc<dyn Fn(Duration) + Send + Sync>,
    write_line: Arc<dyn Fn(&str) + Send + Sync>,
    #[cfg(test)]
    signal: Option<Arc<TestSignal>>,
}

impl Hooks {
    fn production() -> Self {
        let origin = Instant::now();
        Self {
            now: Arc::new(move || origin.elapsed()),
            sleep: Arc::new(thread::sleep),
            write_line: Arc::new(|line| {
                use std::io::Write;
                let _ = writeln!(std::io::stderr().lock(), "{line}");
            }),
            #[cfg(test)]
            signal: None,
        }
    }
}

#[cfg(test)]
#[derive(Default)]
struct TestSignal {
    state: std::sync::Mutex<(u32, u64, u64)>,
    changed: std::sync::Condvar,
}

#[cfg(test)]
impl TestSignal {
    /// Called for every frame the worker pops: processed, or discarded as stale/late.
    fn disposed(&self, epoch: u32, index: u64) {
        if let Ok(mut state) = self.state.lock() {
            if state.0 != epoch {
                *state = (epoch, index, state.2);
            } else {
                state.1 = state.1.max(index);
            }
            self.changed.notify_all();
        }
    }

    fn parked(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.2 = state.2.wrapping_add(1);
            self.changed.notify_all();
        }
    }

    fn wait_processed(&self, epoch: u32, index: u64) {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (state, result) = self
            .changed
            .wait_timeout_while(state, Duration::from_secs(5), |state| {
                state.0 != epoch || state.1 < index
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            !result.timed_out(),
            "worker did not process epoch {epoch} frame {index}"
        );
        assert_eq!(state.0, epoch);
        assert!(state.1 >= index);
    }

    fn wait_parked(&self, previous: u64) {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (state, result) = self
            .changed
            .wait_timeout_while(state, Duration::from_secs(5), |state| state.2 <= previous)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(!result.timed_out(), "worker did not park");
        assert!(state.2 > previous);
    }
}

struct Ready {
    hop: usize,
    delay: usize,
}

/// Fixed-timeline mono engine. Frame k contains input samples `[k*h, (k+1)*h)`;
/// its backend output is rendered at `[k*h + L-D, (k+1)*h + L-D)`.
pub struct Engine {
    input: Producer<Frame>,
    output: Consumer<Frame>,
    worker: Option<JoinHandle<()>>,
    worker_thread: Thread,
    shared: Arc<Shared>,
    hop: usize,
    delay: usize,
    dry: Vec<f32>,
    staging: Frame,
    staging_fill: usize,
    input_frame: u64,
    sample_index: u64,
    latency: Option<usize>,
    current: Option<Frame>,
    next: Option<Frame>,
    render_frame: Option<u64>,
    mix: f32,
    ramp_from: f32,
    ramp_target: f32,
    ramp_position: usize,
    tail_dry: bool,
    block_force_dry: bool,
    frame_wet: bool,
    frame_concealed: bool,
    #[cfg(test)]
    drop_next_input: bool,
    #[cfg(test)]
    last_submitted: Option<(u32, u64)>,
}

impl Engine {
    pub fn spawn(label: &'static str, factory: BackendFactory) -> Result<Self, String> {
        Self::spawn_with_hooks(label, factory, Hooks::production())
    }

    fn spawn_with_hooks(
        label: &'static str,
        factory: BackendFactory,
        hooks: Hooks,
    ) -> Result<Self, String> {
        let (input, worker_input) = RingBuffer::new(FRAME_CAPACITY);
        let (worker_output, output) = RingBuffer::new(FRAME_CAPACITY);
        let shared = Arc::new(Shared::new());
        let worker_shared = Arc::clone(&shared);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let worker_hooks = hooks;
        let spawn_logger = Arc::clone(&worker_hooks.write_line);
        let worker_label = label.to_owned();
        let worker = thread::Builder::new()
            .name(INIT_THREAD_NAME.to_owned())
            .spawn(move || {
                worker_entry(
                    worker_label,
                    factory,
                    worker_input,
                    worker_output,
                    worker_shared,
                    ready_tx,
                    worker_hooks,
                );
            });
        let worker = match worker {
            Ok(worker) => worker,
            Err(error) => {
                let reason = format!("could not spawn worker: {error}");
                spawn_logger(&format!(
                    "clearvoice-ladspa fatal label={label} reason={reason}"
                ));
                return Err(reason);
            }
        };
        let worker_thread = worker.thread().clone();
        let ready = match ready_rx.recv() {
            Ok(Ok(ready)) => ready,
            Ok(Err(error)) => {
                let _ = worker.join();
                return Err(error);
            }
            Err(error) => {
                shared.dead.store(true, Ordering::Release);
                let _ = worker.join();
                return Err(format!("worker exited before ready: {error}"));
            }
        };

        Ok(Self {
            input,
            output,
            worker: Some(worker),
            worker_thread,
            shared,
            hop: ready.hop,
            delay: ready.delay,
            dry: vec![0.0; DRY_CAPACITY],
            staging: Frame::empty(0, 0),
            staging_fill: 0,
            input_frame: 0,
            sample_index: 0,
            latency: None,
            current: None,
            next: None,
            render_frame: None,
            mix: 0.0,
            ramp_from: 0.0,
            ramp_target: 0.0,
            ramp_position: FADE_SAMPLES,
            tail_dry: false,
            block_force_dry: false,
            frame_wet: false,
            frame_concealed: false,
            #[cfg(test)]
            drop_next_input: false,
            #[cfg(test)]
            last_submitted: None,
        })
    }

    pub fn effective_latency(&self) -> Option<usize> {
        self.latency
    }

    /// Latches the first requested target; later calls reset timeline state but retain L.
    pub fn activate(&mut self, latency_ms: f32) -> usize {
        let latency = *self.latency.get_or_insert_with(|| {
            let requested = if latency_ms.is_finite() {
                latency_ms
            } else {
                DEFAULT_LATENCY_MS
            };
            let min = self.delay + self.hop + FADE_SAMPLES + 63;
            ((requested.clamp(10.0, 200.0) * 48.0).round() as usize).clamp(min, MAX_LATENCY)
        });
        let epoch = self
            .shared
            .epoch
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        self.shared.needed_frame.store(0, Ordering::Release);
        self.staging = Frame::empty(epoch, 0);
        self.staging_fill = 0;
        self.input_frame = 0;
        self.sample_index = 0;
        self.current = None;
        self.next = None;
        self.render_frame = None;
        self.mix = 0.0;
        self.ramp_from = 0.0;
        self.ramp_target = 0.0;
        self.ramp_position = FADE_SAMPLES;
        self.tail_dry = false;
        self.block_force_dry = false;
        self.frame_wet = false;
        self.frame_concealed = false;
        #[cfg(test)]
        {
            self.drop_next_input = false;
        }
        latency
    }

    pub fn deactivate(&mut self) {}

    pub fn set_bypass(&self, bypass: bool) {
        self.shared.bypass.store(bypass, Ordering::Release);
    }

    pub fn set_backend_control(&self, index: usize, value: f32) {
        if let Some(control) = self.shared.controls.get(index) {
            control.store(value.to_bits(), Ordering::Release);
        }
    }

    pub fn stats(&self) -> EngineStats {
        self.shared.counters.snapshot()
    }

    #[cfg(test)]
    fn run(&mut self, input: &[f32], output: &mut [f32]) {
        let count = input.len().min(output.len());
        // SAFETY: both slices hold at least `count` samples.
        unsafe { self.run_raw(input.as_ptr(), output.as_mut_ptr(), count) };
    }

    /// Run one complete LADSPA callback. Input and output may alias.
    pub(crate) unsafe fn run_raw(&mut self, input: *const f32, output: *mut f32, count: usize) {
        let slack = self.worker_slack();
        self.block_force_dry = self.latency.is_some() && count > slack;
        if self.block_force_dry {
            self.shared
                .counters
                .oversized_blocks
                .fetch_add(1, Ordering::Relaxed);
        }
        for index in 0..count {
            let sample = if input.is_null() {
                0.0
            } else {
                // SAFETY: the LADSPA host provides `count` readable samples when non-null.
                unsafe { input.add(index).read() }
            };
            let result = self.process_sample_inner(sample);
            if !output.is_null() {
                // SAFETY: the LADSPA host provides `count` writable samples when non-null.
                unsafe { output.add(index).write(result) };
            }
        }
        self.block_force_dry = false;
    }

    fn worker_slack(&self) -> usize {
        self.latency
            .map(|latency| latency.saturating_sub(self.delay + self.hop + FADE_SAMPLES) + 1)
            .unwrap_or(0)
    }

    fn process_sample_inner(&mut self, input: f32) -> f32 {
        let sample = if input.is_finite() { input } else { 0.0 };
        let now = self.sample_index;
        if let Some(slot) = self.dry.get_mut((now % DRY_CAPACITY as u64) as usize) {
            *slot = sample;
        }
        let dry = self
            .latency
            .and_then(|latency| {
                (now >= latency as u64)
                    .then(|| {
                        self.dry
                            .get(((now - latency as u64) % DRY_CAPACITY as u64) as usize)
                    })
                    .flatten()
                    .copied()
            })
            .unwrap_or(0.0);

        let epoch = self.shared.epoch.load(Ordering::Acquire);
        if let Some(slot) = self.staging.samples.get_mut(self.staging_fill) {
            *slot = sample;
        }
        self.staging_fill += 1;
        if self.staging_fill == self.hop {
            self.staging.epoch = epoch;
            self.staging.index = self.input_frame;
            self.staging_fill = 0;
            self.input_frame = self.input_frame.wrapping_add(1);
            let staged =
                std::mem::replace(&mut self.staging, Frame::empty(epoch, self.input_frame));
            #[cfg(test)]
            let forced_drop = std::mem::take(&mut self.drop_next_input);
            #[cfg(not(test))]
            let forced_drop = false;
            #[cfg(test)]
            let submitted = (staged.epoch, staged.index);
            if forced_drop || self.input.push(staged).is_err() {
                self.shared
                    .counters
                    .input_dropped
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                #[cfg(test)]
                {
                    self.last_submitted = Some(submitted);
                }
                self.worker_thread.unpark();
            }
        }

        let output = if let Some(latency) = self.latency {
            let base = latency.saturating_sub(self.delay) as u64;
            if now >= base {
                let relative = now - base;
                let frame = relative / self.hop as u64;
                let offset = (relative % self.hop as u64) as usize;
                self.shared.needed_frame.store(frame, Ordering::Release);
                let new_frame = self.render_frame != Some(frame);
                self.prepare_output_frame(epoch, frame);
                let bypass = self.shared.bypass.load(Ordering::Acquire);
                if new_frame {
                    self.frame_wet = self.current.as_ref().is_some_and(|candidate| {
                        candidate.epoch == epoch && candidate.index == frame && !candidate.settling
                    });
                    self.start_ramp(
                        if self.frame_wet && !bypass && !self.block_force_dry && !self.is_dead() {
                            1.0
                        } else {
                            0.0
                        },
                    );
                    self.tail_dry = false;
                    self.frame_concealed = false;
                    self.render_frame = Some(frame);
                }
                if offset == self.hop.saturating_sub(FADE_SAMPLES) {
                    let next_ready = self.next.as_ref().is_some_and(|candidate| {
                        candidate.epoch == epoch
                            && candidate.index == frame.wrapping_add(1)
                            && !candidate.settling
                    });
                    self.tail_dry = !next_ready || bypass || self.block_force_dry || self.is_dead();
                    if self.tail_dry {
                        self.start_ramp(0.0);
                    }
                }
                // Bypass and worker death ramp out over wet samples already in hand: a ramp
                // started after hop - F continues into the next frame, which the tail check
                // at hop - F guaranteed was in hand. Frames published before a death are valid.
                let dead = self.is_dead();
                let counted = !bypass && !dead;
                let desired =
                    self.frame_wet && !bypass && !self.block_force_dry && !self.tail_dry && !dead;
                if desired && self.ramp_target != 1.0 {
                    self.start_ramp(1.0);
                } else if !desired && !self.tail_dry {
                    self.start_ramp(0.0);
                }
                self.advance_ramp();
                let use_wet = self.frame_wet && self.mix > 0.0;
                if use_wet {
                    if self.mix < 1.0 && counted {
                        self.mark_concealed();
                    }
                    let wet = self
                        .current
                        .as_ref()
                        .and_then(|candidate| candidate.samples.get(offset))
                        .copied()
                        .filter(|value| value.is_finite())
                        .unwrap_or(0.0);
                    wet * self.mix + dry * (1.0 - self.mix)
                } else {
                    if counted {
                        self.mark_concealed();
                    }
                    dry
                }
            } else {
                dry
            }
        } else {
            0.0
        };
        self.sample_index = now.wrapping_add(1);
        output
    }

    fn prepare_output_frame(&mut self, epoch: u32, frame: u64) {
        if self.render_frame != Some(frame) {
            if self
                .next
                .as_ref()
                .is_some_and(|next| next.epoch == epoch && next.index == frame)
            {
                self.current = self.next.take();
            } else if !self
                .current
                .as_ref()
                .is_some_and(|current| current.epoch == epoch && current.index == frame)
            {
                self.current = None;
            }
            if self
                .next
                .as_ref()
                .is_some_and(|next| next.epoch != epoch || next.index != frame.wrapping_add(1))
            {
                self.next = None;
            }
        }
        loop {
            let peeked = self.output.peek();
            let (candidate_epoch, candidate_index) = match peeked {
                Ok(candidate) => (candidate.epoch, candidate.index),
                Err(_) => break,
            };
            if candidate_epoch != epoch || candidate_index < frame {
                let popped = self.output.pop();
                if popped.is_ok() {
                    self.shared
                        .counters
                        .discarded_late
                        .fetch_add(1, Ordering::Relaxed);
                }
                continue;
            }
            if candidate_index > frame.wrapping_add(1) {
                break;
            }
            let candidate = match self.output.pop() {
                Ok(candidate) => candidate,
                Err(_) => break,
            };
            if candidate.index == frame {
                self.current = Some(candidate);
            } else {
                self.next = Some(candidate);
            }
        }
    }

    fn start_ramp(&mut self, target: f32) {
        if self.ramp_target != target || self.ramp_position >= FADE_SAMPLES {
            self.ramp_from = self.mix;
            self.ramp_target = target;
            self.ramp_position = 0;
        }
    }

    fn mark_concealed(&mut self) {
        if !self.frame_concealed {
            self.frame_concealed = true;
            self.shared
                .counters
                .concealed
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn advance_ramp(&mut self) {
        if self.ramp_position < FADE_SAMPLES {
            let fraction = self.ramp_position as f32 / (FADE_SAMPLES - 1) as f32;
            self.mix = self.ramp_from + (self.ramp_target - self.ramp_from) * fraction;
            self.ramp_position += 1;
        } else {
            self.mix = self.ramp_target;
        }
    }

    fn is_dead(&self) -> bool {
        self.shared.dead.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn worker_epoch(&self) -> u32 {
        self.shared.epoch.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn wait_processed(&self, epoch: u32, index: u64, signal: &TestSignal) {
        signal.wait_processed(epoch, index);
    }

    #[cfg(test)]
    fn reset_test_counters(&self) {
        self.shared.counters.processed.store(0, Ordering::Relaxed);
        self.shared.counters.concealed.store(0, Ordering::Relaxed);
        self.shared
            .counters
            .discarded_late
            .store(0, Ordering::Relaxed);
        self.shared
            .counters
            .input_dropped
            .store(0, Ordering::Relaxed);
        self.shared
            .counters
            .output_dropped
            .store(0, Ordering::Relaxed);
        self.shared.counters.settling.store(0, Ordering::Relaxed);
        self.shared.counters.throttled.store(0, Ordering::Relaxed);
        self.shared
            .counters
            .oversized_blocks
            .store(0, Ordering::Relaxed);
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.worker_thread.unpark();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn worker_entry(
    label: String,
    factory: BackendFactory,
    mut input: Consumer<Frame>,
    mut output: Producer<Frame>,
    shared: Arc<Shared>,
    ready: mpsc::SyncSender<Result<Ready, String>>,
    hooks: Hooks,
) {
    install_worker_panic_hook();
    let constructed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(factory));
    let mut backend = match constructed {
        Ok(Ok(backend)) => backend,
        Ok(Err(reason)) => {
            fatal(&label, &reason, &shared, &hooks);
            let _ = ready.send(Err(reason));
            return;
        }
        Err(payload) => {
            let reason = panic_reason(payload);
            fatal(&label, &reason, &shared, &hooks);
            let _ = ready.send(Err(reason));
            return;
        }
    };
    let geometry = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (backend.hop(), backend.delay(), backend.settle_frames())
    }));
    let (hop, delay, settle_frames) = match geometry {
        Ok(geometry) => geometry,
        Err(payload) => {
            let reason = panic_reason(payload);
            fatal(&label, &reason, &shared, &hooks);
            let _ = ready.send(Err(reason));
            return;
        }
    };
    if hop == 0
        || hop > FRAME_SAMPLES
        || delay
            .saturating_add(hop)
            .saturating_add(FADE_SAMPLES)
            .saturating_add(63)
            > MAX_LATENCY
    {
        let reason = format!("unsupported backend geometry hop={hop} delay={delay}");
        fatal(&label, &reason, &shared, &hooks);
        let _ = ready.send(Err(reason));
        return;
    }
    let silence = [0.0; FRAME_SAMPLES];
    let mut warm_output = [0.0; FRAME_SAMPLES];
    let warm_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for _ in 0..settle_frames {
            backend.process(&silence[..hop], &mut warm_output[..hop])?;
        }
        backend.reset();
        Ok::<(), String>(())
    }));
    match warm_result {
        Ok(Ok(())) => {}
        Ok(Err(reason)) => {
            fatal(&label, &reason, &shared, &hooks);
            let _ = ready.send(Err(reason));
            return;
        }
        Err(payload) => {
            let reason = panic_reason(payload);
            fatal(&label, &reason, &shared, &hooks);
            let _ = ready.send(Err(reason));
            return;
        }
    }
    // ClearVoice promotes the thread named cv-dsp-worker to SCHED_RR. Model construction is
    // one long unthrottled call, so the name appears only now, when the worker loop's nap
    // throttle bounds RT runtime. If renaming fails the thread is never promoted (safe).
    let _ = std::fs::write("/proc/thread-self/comm", WORKER_THREAD_NAME);
    if ready.send(Ok(Ready { hop, delay })).is_err() {
        return;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        worker_loop(
            &label,
            &mut *backend,
            &mut input,
            &mut output,
            &shared,
            &hooks,
        )
    }));
    match result {
        Ok(Ok(())) => {}
        Ok(Err(reason)) => fatal(&label, &reason, &shared, &hooks),
        Err(payload) => fatal(&label, &panic_reason(payload), &shared, &hooks),
    }
}

fn worker_loop(
    label: &str,
    backend: &mut dyn Backend,
    input: &mut Consumer<Frame>,
    output: &mut Producer<Frame>,
    shared: &Shared,
    hooks: &Hooks,
) -> Result<(), String> {
    let mut busy_since = (hooks.now)();
    let mut last_stats = busy_since;
    let mut worker_epoch = None;
    let mut expected_index = None;
    let mut reset_pending = false;
    let mut settle_remaining = 0usize;
    let mut process_output = [0.0; FRAME_SAMPLES];

    loop {
        if shared.stop.load(Ordering::Acquire) {
            return Ok(());
        }
        let frame = match input.pop() {
            Ok(frame) => frame,
            Err(_) => {
                let now = (hooks.now)();
                if now.saturating_sub(last_stats) >= STATS_INTERVAL {
                    let counters = shared.counters.snapshot();
                    let line = format!(
                        "clearvoice-ladspa stats label={label} processed={} concealed={} discarded_late={} input_dropped={} output_dropped={} settling={} throttled={} oversized_blocks={}",
                        counters.processed,
                        counters.concealed,
                        counters.discarded_late,
                        counters.input_dropped,
                        counters.output_dropped,
                        counters.settling,
                        counters.throttled,
                        counters.oversized_blocks,
                    );
                    (hooks.write_line)(&line);
                    last_stats = now;
                }
                #[cfg(test)]
                if let Some(signal) = &hooks.signal {
                    signal.parked();
                }
                // A push after the failed pop leaves an unpark token, so no wakeup is lost.
                thread::park();
                continue;
            }
        };
        // RLIMIT_RTTIME counts scheduler ticks while running RT since the last wakeup from
        // a real block. Parks are not trusted as blocks (a pending token returns at once),
        // so the budget is wall time since the last explicit nap: ticks can never exceed
        // elapsed wall time, giving a hard bound of 50 ms + one inference, well under
        // rt.time.soft = 150 ms. Cost: a 1 ms nap per 50 ms (also while mostly idle).
        if (hooks.now)().saturating_sub(busy_since) >= THROTTLE_AFTER {
            (hooks.sleep)(THROTTLE_SLEEP);
            shared.counters.throttled.fetch_add(1, Ordering::Relaxed);
            busy_since = (hooks.now)();
        }
        if let Some(epoch) = worker_epoch {
            if frame.epoch != epoch {
                reset_pending = true;
            }
        } else {
            reset_pending = true;
        }
        if expected_index.is_some_and(|expected| frame.index != expected) {
            reset_pending = true;
        }
        if frame.epoch != shared.epoch.load(Ordering::Acquire)
            || frame.index < shared.needed_frame.load(Ordering::Acquire)
        {
            shared
                .counters
                .discarded_late
                .fetch_add(1, Ordering::Relaxed);
            reset_pending = true;
            worker_epoch = Some(frame.epoch);
            expected_index = Some(frame.index.wrapping_add(1));
            #[cfg(test)]
            if let Some(signal) = &hooks.signal {
                signal.disposed(frame.epoch, frame.index);
            }
            continue;
        }
        if reset_pending {
            backend.reset();
            settle_remaining = backend.settle_frames();
            reset_pending = false;
        }
        for (index, control) in shared.controls.iter().enumerate() {
            let value = f32::from_bits(control.load(Ordering::Acquire));
            backend.set_control(index, value);
        }
        let process = backend.process(
            &frame.samples[..backend.hop()],
            &mut process_output[..backend.hop()],
        );
        process?;
        shared.counters.processed.fetch_add(1, Ordering::Relaxed);
        let mut result = Frame::empty(frame.epoch, frame.index);
        // Bypass is applied only on the run side, per rendered frame, so it never mislabels
        // a frame processed under different controls.
        result.settling = settle_remaining > 0;
        result.samples[..backend.hop()].copy_from_slice(&process_output[..backend.hop()]);
        for sample in &mut result.samples[..backend.hop()] {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
        if settle_remaining > 0 {
            settle_remaining -= 1;
            shared.counters.settling.fetch_add(1, Ordering::Relaxed);
        }
        worker_epoch = Some(frame.epoch);
        expected_index = Some(frame.index.wrapping_add(1));
        if frame.epoch != shared.epoch.load(Ordering::Acquire) {
            shared
                .counters
                .discarded_late
                .fetch_add(1, Ordering::Relaxed);
            reset_pending = true;
        } else if output.push(result).is_err() {
            shared
                .counters
                .output_dropped
                .fetch_add(1, Ordering::Relaxed);
            reset_pending = true;
        }
        #[cfg(test)]
        if let Some(signal) = &hooks.signal {
            signal.disposed(frame.epoch, frame.index);
        }
    }
}

fn fatal(label: &str, reason: &str, shared: &Shared, hooks: &Hooks) {
    shared.dead.store(true, Ordering::Release);
    let reason = reason.replace(['\n', '\r'], " ");
    (hooks.write_line)(&format!(
        "clearvoice-ladspa fatal label={label} reason={reason}"
    ));
}

fn panic_reason(payload: Box<dyn Any + Send>) -> String {
    if let Some(reason) = payload.downcast_ref::<String>() {
        reason.clone()
    } else if let Some(reason) = payload.downcast_ref::<&'static str>() {
        (*reason).to_owned()
    } else {
        "non-string panic payload".to_owned()
    }
}

fn install_worker_panic_hook() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let host_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // The Rust-side name stays INIT_THREAD_NAME after the kernel rename.
            let current = thread::current();
            if current.name() != Some(INIT_THREAD_NAME) {
                host_hook(info);
            }
        }));
    });
}

#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::mpsc::Receiver;

    struct DelayedIdentity {
        hop: usize,
        gain: f32,
        history: [f32; FRAME_SAMPLES],
        gate: Option<Arc<Gate>>,
        gate_used: Arc<AtomicBool>,
        runtime: bool,
    }

    impl DelayedIdentity {
        fn new(hop: usize, gain: f32) -> Self {
            Self {
                hop,
                gain,
                history: [0.0; FRAME_SAMPLES],
                gate: None,
                gate_used: Arc::new(AtomicBool::new(false)),
                runtime: false,
            }
        }

        fn gated(hop: usize, gain: f32, gate: Arc<Gate>) -> Self {
            let mut backend = Self::new(hop, gain);
            backend.gate = Some(gate);
            backend
        }
    }

    impl Backend for DelayedIdentity {
        fn hop(&self) -> usize {
            self.hop
        }
        fn delay(&self) -> usize {
            self.hop
        }
        fn settle_frames(&self) -> usize {
            1
        }

        fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
            if let Some(gate) = self
                .gate
                .as_ref()
                .filter(|_| self.runtime && !self.gate_used.swap(true, Ordering::AcqRel))
            {
                gate.block();
            }
            for (index, value) in output.iter_mut().enumerate() {
                *value = self.history.get(index).copied().unwrap_or(0.0) * self.gain;
            }
            if let Some(history) = self.history.get_mut(..input.len()) {
                history.copy_from_slice(input);
            }
            Ok(())
        }

        fn reset(&mut self) {
            self.history.fill(0.0);
            self.runtime = true;
        }
    }

    struct ConstantBackend {
        hop: usize,
        value: f32,
    }

    impl Backend for ConstantBackend {
        fn hop(&self) -> usize {
            self.hop
        }
        fn delay(&self) -> usize {
            self.hop
        }
        fn settle_frames(&self) -> usize {
            1
        }
        fn process(&mut self, _input: &[f32], output: &mut [f32]) -> Result<(), String> {
            output.fill(self.value);
            Ok(())
        }
        fn reset(&mut self) {}
    }

    struct StatefulBackend {
        hop: usize,
        sum: f32,
    }

    impl Backend for StatefulBackend {
        fn hop(&self) -> usize {
            self.hop
        }
        fn delay(&self) -> usize {
            0
        }
        fn settle_frames(&self) -> usize {
            2
        }
        fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
            for (source, target) in input.iter().zip(output.iter_mut()) {
                self.sum += *source;
                *target = self.sum;
            }
            Ok(())
        }
        fn reset(&mut self) {
            self.sum = 0.0;
        }
    }

    struct FailingBackend {
        armed: bool,
        panic: bool,
    }

    impl Backend for FailingBackend {
        fn hop(&self) -> usize {
            480
        }
        fn delay(&self) -> usize {
            480
        }
        fn settle_frames(&self) -> usize {
            1
        }
        fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
            if self.armed {
                if self.panic {
                    panic!("process exploded");
                }
                return Err("process exploded".to_owned());
            }
            output.copy_from_slice(input);
            Ok(())
        }
        fn reset(&mut self) {
            self.armed = true;
        }
    }

    struct Gate {
        state: std::sync::Mutex<(bool, bool)>,
        changed: std::sync::Condvar,
    }

    impl Gate {
        fn new() -> Self {
            Self {
                state: std::sync::Mutex::new((false, false)),
                changed: std::sync::Condvar::new(),
            }
        }

        fn block(&self) {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.0 = true;
            self.changed.notify_all();
            let (state, result) = self
                .changed
                .wait_timeout_while(state, Duration::from_secs(5), |s| !s.1)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(!result.timed_out(), "test gate was not released");
            assert!(state.1);
        }

        fn wait_entered(&self) {
            let state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let (state, result) = self
                .changed
                .wait_timeout_while(state, Duration::from_secs(5), |s| !s.0)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(!result.timed_out(), "worker did not enter test gate");
            assert!(state.0);
        }

        fn release(&self) {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.1 = true;
            self.changed.notify_all();
        }
    }

    fn hooks(
        signal: Arc<TestSignal>,
        log: std::sync::mpsc::Sender<String>,
        now: Arc<dyn Fn() -> Duration + Send + Sync>,
        sleep: Arc<dyn Fn(Duration) + Send + Sync>,
    ) -> Hooks {
        Hooks {
            now,
            sleep,
            write_line: Arc::new(move |line| {
                let _ = log.send(line.to_owned());
            }),
            signal: Some(signal),
        }
    }

    fn spawn_backend<B, F>(make: F) -> (Engine, Arc<TestSignal>, Receiver<String>)
    where
        B: Backend,
        F: FnOnce() -> B + Send + 'static,
    {
        let signal = Arc::new(TestSignal::default());
        let (log_tx, log_rx) = mpsc::channel();
        let origin = Instant::now();
        let hooks = hooks(
            Arc::clone(&signal),
            log_tx,
            Arc::new(move || origin.elapsed()),
            Arc::new(thread::sleep),
        );
        let factory: BackendFactory = Box::new(move || Ok(Box::new(make()) as Box<dyn Backend>));
        let engine = Engine::spawn_with_hooks("test-fake", factory, hooks).expect("engine startup");
        (engine, signal, log_rx)
    }

    /// Runs one callback, then waits until the worker has disposed of the last frame this
    /// callback actually submitted (frames that failed to push are never waited for).
    fn run_and_wait(
        engine: &mut Engine,
        signal: &TestSignal,
        _epoch: u32,
        input: &[f32],
        output: &mut [f32],
    ) {
        let before = engine.last_submitted;
        engine.run(input, output);
        if let Some((epoch, index)) = engine
            .last_submitted
            .filter(|_| engine.last_submitted != before)
        {
            engine.wait_processed(epoch, index, signal);
        }
    }

    fn feed_quantum(
        engine: &mut Engine,
        signal: &TestSignal,
        epoch: u32,
        input: &[f32],
        output: &mut [f32],
        quantum: usize,
    ) {
        let mut start = 0;
        while start < input.len() {
            let end = (start + quantum).min(input.len());
            run_and_wait(
                engine,
                signal,
                epoch,
                &input[start..end],
                &mut output[start..end],
            );
            start = end;
        }
    }

    /// Impulse after three primed frames (settle + fade-in done). With the gain-2 delay-D fake,
    /// a wet render is exactly 2.0 at L; an aligned-dry render is exactly 1.0 at L.
    fn check_impulse(
        engine: &mut Engine,
        signal: &TestSignal,
        hop: usize,
        quantum: usize,
        latency_ms: f32,
        phase: usize,
    ) {
        let latency = engine.activate(latency_ms);
        let expected_latency = if latency_ms == 10.0 {
            hop * 2 + FADE_SAMPLES + 63
        } else {
            (latency_ms * 48.0).round() as usize
        };
        assert_eq!(latency, expected_latency);
        let at = 3 * hop + phase;
        // Whole callbacks only, so an oversized quantum never ends with a small (wet) tail block.
        let mut input = vec![0.0; (at + latency + 1).div_ceil(quantum) * quantum];
        let mut output = vec![0.0; input.len()];
        input[at] = 1.0;
        let epoch = engine.worker_epoch();
        engine.reset_test_counters();
        feed_quantum(engine, signal, epoch, &input, &mut output, quantum);
        let expected = at + latency;
        let wet = quantum <= engine.worker_slack();
        assert_eq!(
            output[expected],
            if wet { 2.0 } else { 1.0 },
            "q={quantum} phase={phase} L={latency}"
        );
        assert!(
            output
                .iter()
                .enumerate()
                .all(|(index, value)| index == expected || value.abs() < 1e-6),
            "stray output q={quantum} phase={phase} L={latency}"
        );
        let stats = engine.stats();
        if wet {
            assert_eq!(stats.oversized_blocks, 0);
        } else {
            assert!(stats.oversized_blocks > 0 && stats.concealed > 0);
        }
        assert_eq!(engine.effective_latency(), Some(latency));
    }

    fn thread_names() -> Vec<String> {
        std::fs::read_dir("/proc/self/task")
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter_map(|entry| std::fs::read_to_string(entry.path().join("comm")).ok())
            .map(|name| name.trim().to_owned())
            .collect()
    }

    fn worker_thread_count() -> usize {
        thread_names()
            .iter()
            .filter(|name| *name == INIT_THREAD_NAME || *name == WORKER_THREAD_NAME)
            .count()
    }

    #[test]
    fn rt_target_name_appears_only_after_backend_construction() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let seen = Arc::new(std::sync::Mutex::new(String::new()));
        let seen_in_factory = Arc::clone(&seen);
        let (mut engine, _, _) = spawn_backend(move || {
            if let Ok(mut name) = seen_in_factory.lock() {
                *name = std::fs::read_to_string("/proc/thread-self/comm")
                    .unwrap_or_default()
                    .trim()
                    .to_owned();
            }
            DelayedIdentity::new(480, 2.0)
        });
        assert_eq!(*seen.lock().expect("name"), INIT_THREAD_NAME);
        assert!(thread_names().iter().any(|name| name == WORKER_THREAD_NAME));
        engine.activate(35.0);
        drop(engine);
    }

    #[test]
    fn impulse_alignment_all_quanta_phases_and_latency_targets() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let quanta = [64, 256, 480, 512, 1024, 2048];
        for latency in [10.0, 35.0, 200.0] {
            for quantum in quanta {
                let (mut engine, signal, _) = spawn_backend(|| DelayedIdentity::new(480, 2.0));
                for phase in 0..quantum {
                    check_impulse(&mut engine, &signal, 480, quantum, latency, phase);
                }
            }
        }
    }

    #[test]
    fn hop_512_and_8192_quantum_impulses_stay_aligned() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (mut engine, signal, _) = spawn_backend(|| DelayedIdentity::new(512, 2.0));
        for quantum in [64, 512, 1024, 2048] {
            for phase in [0, quantum - 1] {
                check_impulse(&mut engine, &signal, 512, quantum, 35.0, phase);
            }
        }
        let (mut long_engine, long_signal, _) = spawn_backend(|| DelayedIdentity::new(512, 2.0));
        for phase in [0, 4096, 8191] {
            check_impulse(&mut long_engine, &long_signal, 512, 8192, 200.0, phase);
        }
        assert_eq!(long_engine.effective_latency(), Some(9600));
    }

    /// Wet gain-2 fake that fails (fatal) on its `life`-th processed frame.
    struct DyingBackend {
        inner: DelayedIdentity,
        life: usize,
    }

    impl Backend for DyingBackend {
        fn hop(&self) -> usize {
            480
        }
        fn delay(&self) -> usize {
            480
        }
        fn settle_frames(&self) -> usize {
            1
        }
        fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
            self.life = self.life.saturating_sub(1);
            if self.life == 0 {
                return Err("died".to_owned());
            }
            self.inner.process(input, output)
        }
        fn reset(&mut self) {
            self.inner.reset();
        }
    }

    /// Constant input 1.0 renders dry 1.0 / wet 2.0, so any switch without a crossfade
    /// shows up as a step larger than one ramp increment.
    fn assert_continuous(output: &[f32]) {
        let step = 1.0 / (FADE_SAMPLES - 1) as f32 + 1e-6;
        for (index, pair) in output.windows(2).enumerate() {
            assert!(
                (pair[1] - pair[0]).abs() <= step,
                "jump at {index}: {pair:?}"
            );
            assert!(
                (1.0..=2.0).contains(&pair[1]),
                "out of range at {index}: {pair:?}"
            );
        }
    }

    #[test]
    fn bypass_entry_and_worker_death_crossfade_to_dry() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Bypass entered late in a wet frame (offset 400 > hop - F) and early (offset 100).
        for offset in [400usize, 100] {
            let (mut engine, signal, _) = spawn_backend(|| DelayedIdentity::new(480, 2.0));
            let latency = engine.activate(35.0);
            let epoch = engine.worker_epoch();
            let base = latency - 480;
            let switch = base + 6 * 480 + offset;
            let input = vec![1.0; switch + 4 * 480];
            let mut output = vec![0.0; input.len()];
            feed_quantum(
                &mut engine,
                &signal,
                epoch,
                &input[..switch],
                &mut output[..switch],
                64,
            );
            assert_eq!(output[switch - 1], 2.0, "wet before bypass");
            engine.set_bypass(true);
            feed_quantum(
                &mut engine,
                &signal,
                epoch,
                &input[switch..],
                &mut output[switch..],
                64,
            );
            assert_continuous(&output[latency..]);
            assert_eq!(*output.last().expect("output"), 1.0);
        }
        // Worker dies on frame index 5 (life counts the engine's warm-up frame too):
        // published wet frames play out, then fade to dry.
        let (mut engine, signal, logs) = spawn_backend(|| DyingBackend {
            inner: DelayedIdentity::new(480, 2.0),
            life: 7,
        });
        let latency = engine.activate(35.0);
        let epoch = engine.worker_epoch();
        let input = vec![1.0; latency + 12 * 480];
        let mut output = vec![0.0; input.len()];
        // Frames 0..4 are processed (acknowledged). Feeding up to 6 hops submits frame 5,
        // which kills the worker while frame 3 (wet, frame 4 in hand) is being rendered;
        // wait for the death before rendering further so the transition is deterministic.
        let healthy = 5 * 480;
        let dying = 6 * 480;
        feed_quantum(
            &mut engine,
            &signal,
            epoch,
            &input[..healthy],
            &mut output[..healthy],
            64,
        );
        engine.run(&input[healthy..dying], &mut output[healthy..dying]);
        assert!(logs.recv_timeout(Duration::from_secs(5)).is_ok());
        assert!(engine.is_dead());
        assert_eq!(
            output[dying - 1],
            2.0,
            "frame 3 is wet when the worker dies"
        );
        for (chunk, out) in input[dying..]
            .chunks(64)
            .zip(output[dying..].chunks_mut(64))
        {
            engine.run(chunk, out);
        }
        assert_continuous(&output[latency..]);
        assert_eq!(*output.last().expect("output"), 1.0);
    }

    #[test]
    fn bypass_release_lands_wet_impulse_exactly_at_l() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (mut engine, signal, _) = spawn_backend(|| DelayedIdentity::new(480, 2.0));
        let latency = engine.activate(35.0);
        let epoch = engine.worker_epoch();
        // Bypassed (attenuation 0): impulse A renders aligned dry. Released before impulse B's
        // frame: B renders wet. Neither may appear early or twice.
        engine.set_bypass(true);
        let a = 3 * 480 + 17;
        let release_at = latency + 8 * 480;
        let b = release_at + 4 * 480 + 33 - latency;
        let mut input = vec![0.0; b + latency + 480];
        input[a] = 1.0;
        input[b] = 1.0;
        let mut output = vec![0.0; input.len()];
        feed_quantum(
            &mut engine,
            &signal,
            epoch,
            &input[..release_at],
            &mut output[..release_at],
            256,
        );
        engine.set_bypass(false);
        feed_quantum(
            &mut engine,
            &signal,
            epoch,
            &input[release_at..],
            &mut output[release_at..],
            256,
        );
        assert_eq!(output[a + latency], 1.0);
        assert_eq!(output[b + latency], 2.0);
        assert!(
            output
                .iter()
                .enumerate()
                .all(|(index, value)| index == a + latency
                    || index == b + latency
                    || *value == 0.0)
        );
    }

    #[test]
    fn latency_latches_nan_defaults_and_reactivation_reports_same_l() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (mut engine, _, _) = spawn_backend(|| DelayedIdentity::new(480, 2.0));
        assert_eq!(engine.activate(f32::NAN), 1680);
        engine.set_backend_control(2, 18.0);
        engine.deactivate();
        assert_eq!(engine.activate(200.0), 1680);
        assert_eq!(engine.effective_latency(), Some(1680));
        assert_eq!(engine.activate(f32::INFINITY), 1680);
    }

    #[test]
    fn backend_controls_are_snapshotted_per_frame_without_sample_sharing() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let seen = Arc::new(AtomicU32::new(0));
        let backend_seen = Arc::clone(&seen);
        let signal = Arc::new(TestSignal::default());
        let (log_tx, _) = mpsc::channel();
        let origin = Instant::now();
        let hooks = hooks(
            Arc::clone(&signal),
            log_tx,
            Arc::new(move || origin.elapsed()),
            Arc::new(thread::sleep),
        );
        let factory: BackendFactory = Box::new(move || {
            Ok(Box::new(ControlBackend { seen: backend_seen }) as Box<dyn Backend>)
        });
        let mut engine =
            Engine::spawn_with_hooks("test-controls", factory, hooks).expect("control engine");
        engine.activate(35.0);
        let epoch = engine.worker_epoch();
        engine.set_backend_control(3, 12.5);
        engine.run(&[0.0; 480], &mut [0.0; 480]);
        engine.wait_processed(epoch, 0, &signal);
        assert_eq!(f32::from_bits(seen.load(Ordering::Acquire)), 12.5);
    }

    struct ControlBackend {
        seen: Arc<AtomicU32>,
    }

    impl Backend for ControlBackend {
        fn hop(&self) -> usize {
            480
        }
        fn delay(&self) -> usize {
            480
        }
        fn settle_frames(&self) -> usize {
            0
        }
        fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
            output.copy_from_slice(input);
            Ok(())
        }
        fn reset(&mut self) {}
        fn set_control(&mut self, index: usize, value: f32) {
            if index == 3 {
                self.seen.store(value.to_bits(), Ordering::Release);
            }
        }
    }

    #[test]
    fn single_missing_frame_resets_state_settles_dry_then_replays_contiguous_history() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (mut engine, signal, _) = spawn_backend(|| StatefulBackend { hop: 480, sum: 0.0 });
        let latency = engine.activate(35.0);
        let epoch = engine.worker_epoch();
        let mut captured = vec![0.0; latency + 12 * 480];
        for frame in 0..(captured.len() / 480) {
            let input = [1.0; 480];
            let mut output = [0.0; 480];
            if frame == 2 {
                engine.drop_next_input = true;
            }
            if frame == 2 {
                engine.run(&input, &mut output);
            } else {
                run_and_wait(&mut engine, &signal, epoch, &input, &mut output);
            }
            let start = frame * 480;
            if let Some(target) = captured.get_mut(start..start + 480) {
                target.copy_from_slice(&output);
            }
        }
        for frame in [3usize, 4] {
            let start = latency + frame * 480;
            assert!(
                captured[start..start + 480]
                    .iter()
                    .all(|sample| (*sample - 1.0).abs() < 1e-6)
            );
        }
        let after_settle = latency + 5 * 480 + 127;
        assert!(
            (captured[after_settle] - (2.0 * 480.0 + 128.0)).abs() < 1e-3,
            "stateful wet replay mismatch: got {}",
            captured[after_settle]
        );
        assert_eq!(engine.effective_latency(), Some(latency));
        assert_eq!(engine.stats().input_dropped, 1);
        assert!(engine.stats().settling >= 4);
    }

    #[test]
    fn crossfade_ramps_survive_callback_splits_and_remain_finite_bounded() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (mut engine, signal, _) = spawn_backend(|| ConstantBackend {
            hop: 480,
            value: 1.0,
        });
        let latency = engine.activate(200.0);
        let epoch = engine.worker_epoch();
        let preroll = latency;
        let mut fed = 0;
        while fed < preroll {
            let count = (512usize).min(preroll - fed);
            let input = vec![0.0; count];
            let mut output = vec![0.0; count];
            run_and_wait(&mut engine, &signal, epoch, &input, &mut output);
            fed += count;
        }
        let mut ramp = Vec::with_capacity(128);
        for count in [7usize, 23, 1, 41, 56] {
            let input = vec![0.0; count];
            let mut output = vec![0.0; count];
            engine.run(&input, &mut output);
            ramp.extend(output);
        }
        assert_eq!(ramp.len(), 128);
        assert!(ramp[0].abs() < 1e-6);
        assert!((ramp[127] - 1.0).abs() < 1e-6);
        assert!(ramp.windows(2).all(|samples| samples[1] >= samples[0]));
        assert!(
            ramp.iter()
                .all(|sample| sample.is_finite() && sample.abs() <= 1.0)
        );
    }

    #[test]
    fn instant_backend_has_no_steady_concealment_until_quantum_exceeds_slack() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for quantum in [64usize, 256, 480, 512, 1024] {
            let (mut engine, signal, _) = spawn_backend(|| DelayedIdentity::new(480, 1.0));
            let latency = engine.activate(35.0);
            let epoch = engine.worker_epoch();
            let stable_at = latency - 480 + 4 * 480;
            let mut fed = 0;
            while fed < stable_at {
                let count = quantum.min(stable_at - fed);
                let input = vec![0.0; count];
                let mut output = vec![0.0; count];
                run_and_wait(&mut engine, &signal, epoch, &input, &mut output);
                fed += count;
            }
            engine.reset_test_counters();
            for _ in 0..8 {
                let input = vec![0.0; quantum];
                let mut output = vec![0.0; quantum];
                run_and_wait(&mut engine, &signal, epoch, &input, &mut output);
            }
            let stats = engine.stats();
            assert_eq!(engine.effective_latency(), Some(1680));
            assert_eq!(stats.output_dropped, 0);
            if quantum <= 512 {
                assert_eq!(stats.concealed, 0, "q={quantum}");
            } else {
                assert!(stats.concealed > 0);
                assert_eq!(stats.oversized_blocks, 8);
            }
        }
    }

    #[test]
    fn a_paused_worker_cannot_publish_an_old_epoch_after_activate() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let gate = Arc::new(Gate::new());
        let gate_for_backend = Arc::clone(&gate);
        let (mut engine, signal, _) =
            spawn_backend(move || DelayedIdentity::gated(480, 1.0, gate_for_backend));
        engine.activate(35.0);
        let old = engine.worker_epoch();
        let old_input = [9.0; 480];
        let mut old_output = [0.0; 480];
        engine.run(&old_input, &mut old_output);
        gate.wait_entered();

        engine.activate(35.0);
        let new = engine.worker_epoch();
        gate.release();
        let mut captured = vec![0.0; 1680 + 4 * 480];
        let mut offset = 0;
        while offset < captured.len() {
            let count = 480.min(captured.len() - offset);
            let input = vec![2.0; count];
            run_and_wait(
                &mut engine,
                &signal,
                new,
                &input,
                &mut captured[offset..offset + count],
            );
            offset += count;
        }
        assert!(
            captured[1680..]
                .iter()
                .all(|sample| (*sample - 2.0).abs() < 1e-6)
        );
        assert!(old != new);
        assert!(engine.stats().discarded_late >= 1);
    }

    #[test]
    fn input_ring_overflow_is_counted_and_dry_impulse_stays_at_fixed_latency() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let gate = Arc::new(Gate::new());
        let gate_for_backend = Arc::clone(&gate);
        let (mut engine, signal, _) =
            spawn_backend(move || DelayedIdentity::gated(480, 2.0, gate_for_backend));
        let latency = engine.activate(200.0);
        let epoch = engine.worker_epoch();
        let signal_at = |index: usize| (index % 17 + 1) as f32 / 17.0;
        let stall = 66 * 480;
        let input = (0..stall).map(signal_at).collect::<Vec<_>>();
        let mut output = vec![0.0; input.len()];
        engine.run(&input, &mut output);
        gate.wait_entered();
        assert!(engine.stats().input_dropped > 0);
        assert!(
            output[latency..]
                .iter()
                .zip(input[..input.len() - latency].iter())
                .all(|(actual, expected)| actual == expected)
        );
        gate.release();

        // Recovery: keep feeding contiguous audio; after the late backlog is discarded and the
        // post-reset settle frame is rendered dry, output is the fake's wet (2x, delay D) at L.
        let recovery = 40 * 480;
        let tail = (stall..stall + recovery).map(signal_at).collect::<Vec<_>>();
        let mut tail_output = vec![0.0; recovery];
        feed_quantum(&mut engine, &signal, epoch, &tail, &mut tail_output, 480);
        let wet_from = recovery - 10 * 480;
        for (offset, actual) in tail_output.iter().enumerate().skip(wet_from) {
            assert_eq!(
                *actual,
                2.0 * signal_at(stall + offset - latency),
                "offset {offset}"
            );
        }
        assert!(engine.stats().discarded_late > 0 && engine.stats().settling > 0);
        assert_eq!(engine.effective_latency(), Some(9600));
    }

    #[test]
    fn output_ring_overflow_drops_and_counts_without_waiting() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let gate = Arc::new(Gate::new());
        let backend_gate = Arc::clone(&gate);
        let (mut engine, signal, _) =
            spawn_backend(move || DelayedIdentity::gated(480, 1.0, backend_gate));
        engine.activate(200.0);
        let epoch = engine.worker_epoch();
        assert!(engine.input.push(Frame::empty(epoch, 0)).is_ok());
        engine.worker_thread.unpark();
        gate.wait_entered();
        for index in 1..=FRAME_CAPACITY as u64 {
            assert!(engine.input.push(Frame::empty(epoch, index)).is_ok());
        }
        assert!(engine.input.push(Frame::empty(epoch, 65)).is_err());
        gate.release();
        engine.wait_processed(epoch, FRAME_CAPACITY as u64, &signal);
        assert_eq!(engine.stats().output_dropped, 1);
    }

    #[test]
    fn busy_worker_throttles_from_virtual_clock_without_breaking_bypass_alignment() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let signal = Arc::new(TestSignal::default());
        let (log_tx, _) = mpsc::channel();
        let clock = Arc::new(AtomicU64::new(0));
        let clock_for_now = Arc::clone(&clock);
        let clock_for_sleep = Arc::clone(&clock);
        let sleeps = Arc::new(AtomicUsize::new(0));
        let sleeps_for_hook = Arc::clone(&sleeps);
        let hooks = hooks(
            Arc::clone(&signal),
            log_tx,
            Arc::new(move || Duration::from_nanos(clock_for_now.load(Ordering::Relaxed))),
            Arc::new(move |duration| {
                sleeps_for_hook.fetch_add(1, AtomicOrdering::Relaxed);
                clock_for_sleep.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
            }),
        );
        let factory_clock = Arc::clone(&clock);
        let gate = Arc::new(Gate::new());
        let backend_gate = Arc::clone(&gate);
        let factory: BackendFactory = Box::new(move || {
            Ok(Box::new(BusyBackend {
                hop: 480,
                clock: factory_clock,
                gate: backend_gate,
                runtime: false,
            }) as Box<dyn Backend>)
        });
        let mut engine =
            Engine::spawn_with_hooks("test-busy", factory, hooks).expect("busy engine");
        let latency = engine.activate(200.0);
        engine.set_bypass(true);
        let epoch = engine.worker_epoch();
        let mut fed = 0;
        while fed < 10 * 480 {
            let count = 480.min(10 * 480 - fed);
            let mut input = vec![0.0; count];
            if fed == 0 {
                input[0] = 1.0;
            }
            let mut output = vec![0.0; count];
            engine.run(&input, &mut output);
            fed += count;
            if fed == 480 {
                gate.wait_entered();
            }
        }
        gate.release();
        signal.wait_processed(epoch, 9);
        assert!(sleeps.load(AtomicOrdering::Relaxed) >= 8);
        let mut captured = vec![0.0; latency + 1 - fed];
        let mut final_output = vec![0.0; captured.len()];
        let input = vec![0.0; captured.len()];
        feed_quantum(&mut engine, &signal, epoch, &input, &mut final_output, 480);
        captured.copy_from_slice(&final_output);
        assert_eq!(captured[latency - fed], 1.0);
        assert!(
            captured
                .iter()
                .enumerate()
                .all(|(index, value)| index == latency - fed || value.abs() < 1e-6)
        );
    }

    struct ClockBackend {
        clock: Arc<AtomicU64>,
    }

    impl Backend for ClockBackend {
        fn hop(&self) -> usize {
            480
        }
        fn delay(&self) -> usize {
            480
        }
        fn settle_frames(&self) -> usize {
            0
        }
        fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
            self.clock.fetch_add(30_000_000, AtomicOrdering::Relaxed);
            output.copy_from_slice(input);
            Ok(())
        }
        fn reset(&mut self) {}
    }

    /// 30 ms (virtual) of work per frame with a real park between frames.
    fn throttles_with_parks() -> Vec<u64> {
        let signal = Arc::new(TestSignal::default());
        let (log_tx, _) = mpsc::channel();
        let clock = Arc::new(AtomicU64::new(0));
        let now_clock = Arc::clone(&clock);
        let sleep_clock = Arc::clone(&clock);
        let hooks = hooks(
            Arc::clone(&signal),
            log_tx,
            Arc::new(move || Duration::from_nanos(now_clock.load(AtomicOrdering::Relaxed))),
            Arc::new(move |duration| {
                sleep_clock.fetch_add(duration.as_nanos() as u64, AtomicOrdering::Relaxed);
            }),
        );
        let backend_clock = Arc::clone(&clock);
        let factory: BackendFactory = Box::new(move || {
            Ok(Box::new(ClockBackend {
                clock: backend_clock,
            }) as Box<dyn Backend>)
        });
        let mut engine =
            Engine::spawn_with_hooks("test-park", factory, hooks).expect("park engine");
        engine.activate(35.0);
        let epoch = engine.worker_epoch();
        (0..3)
            .map(|_| {
                let parked = signal.state.lock().map(|state| state.2).unwrap_or(0);
                run_and_wait(&mut engine, &signal, epoch, &[0.0; 480], &mut [0.0; 480]);
                signal.wait_parked(parked);
                engine.stats().throttled
            })
            .collect()
    }

    #[test]
    fn throttle_budget_is_wall_time_since_the_last_explicit_nap() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Parks never reset the budget: after 2 x 30 ms the third frame naps first.
        assert_eq!(throttles_with_parks(), [0, 0, 1]);
    }

    #[test]
    fn stats_line_keeps_the_slice_c_field_order_and_names() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let signal = Arc::new(TestSignal::default());
        let (log_tx, log_rx) = mpsc::channel();
        let clock = Arc::new(AtomicU64::new(0));
        let worker_clock = Arc::clone(&clock);
        let hooks = hooks(
            Arc::clone(&signal),
            log_tx,
            Arc::new(move || Duration::from_nanos(worker_clock.load(AtomicOrdering::Relaxed))),
            Arc::new(|_| {}),
        );
        let backend_clock = Arc::clone(&clock);
        let factory: BackendFactory = Box::new(move || {
            Ok(Box::new(StatsBackend {
                clock: backend_clock,
            }) as Box<dyn Backend>)
        });
        let mut engine =
            Engine::spawn_with_hooks("test-stats", factory, hooks).expect("stats engine");
        engine.activate(35.0);
        let epoch = engine.worker_epoch();
        let input = [0.0; 480];
        let mut output = [0.0; 480];
        engine.run(&input, &mut output);
        engine.wait_processed(epoch, 0, &signal);
        assert_eq!(
            log_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("stats line"),
            "clearvoice-ladspa stats label=test-stats processed=1 concealed=0 discarded_late=0 input_dropped=0 output_dropped=0 settling=0 throttled=0 oversized_blocks=0",
        );
    }

    struct StatsBackend {
        clock: Arc<AtomicU64>,
    }

    impl Backend for StatsBackend {
        fn hop(&self) -> usize {
            480
        }
        fn delay(&self) -> usize {
            480
        }
        fn settle_frames(&self) -> usize {
            0
        }
        fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
            output.copy_from_slice(input);
            self.clock.store(10_000_000_000, AtomicOrdering::Relaxed);
            Ok(())
        }
        fn reset(&mut self) {}
    }

    struct BusyBackend {
        hop: usize,
        clock: Arc<AtomicU64>,
        gate: Arc<Gate>,
        runtime: bool,
    }

    impl Backend for BusyBackend {
        fn hop(&self) -> usize {
            self.hop
        }
        fn delay(&self) -> usize {
            self.hop
        }
        fn settle_frames(&self) -> usize {
            0
        }
        fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
            if self.runtime {
                if self.clock.load(AtomicOrdering::Relaxed) == 0 {
                    self.gate.block();
                }
                self.clock.fetch_add(51_000_000, AtomicOrdering::Relaxed);
            }
            output.copy_from_slice(input);
            Ok(())
        }
        fn reset(&mut self) {
            self.runtime = true;
        }
    }

    #[test]
    fn backend_failures_log_once_fall_back_to_dry_and_join() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let before = worker_thread_count();
        let signal = Arc::new(TestSignal::default());
        let (log_tx, log_rx) = mpsc::channel();
        let construction_hooks = hooks(
            Arc::clone(&signal),
            log_tx,
            Arc::new(|| Duration::ZERO),
            Arc::new(|_| {}),
        );
        let factory: BackendFactory = Box::new(|| Err("constructor exploded".to_owned()));
        assert!(
            Engine::spawn_with_hooks("test-construction", factory, construction_hooks).is_err()
        );
        assert_eq!(
            log_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
            "clearvoice-ladspa fatal label=test-construction reason=constructor exploded"
        );
        let signal = Arc::new(TestSignal::default());
        let (log_tx, log_rx) = mpsc::channel();
        let hooks = hooks(
            signal,
            log_tx,
            Arc::new(|| Duration::ZERO),
            Arc::new(|_| {}),
        );
        let factory: BackendFactory = Box::new(|| panic!("constructor panic"));
        assert!(Engine::spawn_with_hooks("test-constructor-panic", factory, hooks).is_err());
        assert_eq!(
            log_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
            "clearvoice-ladspa fatal label=test-constructor-panic reason=constructor panic"
        );
        for panic in [false, true] {
            let (mut engine, signal, logs) = spawn_backend(move || FailingBackend {
                armed: false,
                panic,
            });
            let latency = engine.activate(35.0);
            let input = [1.0; 480];
            let mut output = [0.0; 480];
            engine.run(&input, &mut output);
            let line = logs
                .recv_timeout(Duration::from_secs(5))
                .expect("fatal line");
            assert!(line.starts_with("clearvoice-ladspa fatal label=test-fake reason="));
            let next_input = vec![0.0; latency + 480];
            let mut next_output = vec![0.0; next_input.len()];
            engine.run(&next_input, &mut next_output);
            assert!(
                next_output[latency - 480..latency]
                    .iter()
                    .all(|sample| (*sample - 1.0).abs() < 1e-6)
            );
            assert!(engine.is_dead());
            drop(engine);
            assert!(logs.try_recv().is_err());
            let _ = signal;
        }
        assert_eq!(worker_thread_count(), before);
    }

    #[test]
    fn repeated_lifecycle_joins_every_named_worker() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let before = worker_thread_count();
        for _ in 0..8 {
            let (mut engine, signal, _) = spawn_backend(|| DelayedIdentity::new(480, 1.0));
            signal.wait_parked(0);
            engine.activate(35.0);
            engine.deactivate();
            drop(engine);
        }
        assert_eq!(worker_thread_count(), before);
    }
}
