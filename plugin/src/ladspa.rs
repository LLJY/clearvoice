use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_ulong};
use std::ptr;

use crate::engine::{BACKEND_CONTROL_SLOTS, BackendFactory, Engine};

pub type LadspaData = f32;
pub type LadspaHandle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PortRangeHint {
    pub hint_descriptor: c_int,
    pub lower_bound: LadspaData,
    pub upper_bound: LadspaData,
}

pub type Instantiate = unsafe extern "C" fn(*const Descriptor, c_ulong) -> LadspaHandle;
pub type ConnectPort = unsafe extern "C" fn(LadspaHandle, c_ulong, *mut LadspaData);
pub type Activate = unsafe extern "C" fn(LadspaHandle);
pub type Run = unsafe extern "C" fn(LadspaHandle, c_ulong);
pub type Deactivate = unsafe extern "C" fn(LadspaHandle);
pub type Cleanup = unsafe extern "C" fn(LadspaHandle);

/// Field order and C types match LADSPA 1.1's `LADSPA_Descriptor`.
#[repr(C)]
pub struct Descriptor {
    pub unique_id: c_ulong,
    pub label: *const c_char,
    pub properties: c_int,
    pub name: *const c_char,
    pub maker: *const c_char,
    pub copyright: *const c_char,
    pub port_count: c_ulong,
    pub port_descriptors: *const c_int,
    pub port_names: *const *const c_char,
    pub port_range_hints: *const PortRangeHint,
    pub implementation_data: *mut c_void,
    pub instantiate: Option<Instantiate>,
    pub connect_port: Option<ConnectPort>,
    pub activate: Option<Activate>,
    pub run: Option<Run>,
    pub run_adding: Option<Run>,
    pub set_run_adding_gain: Option<unsafe extern "C" fn(LadspaHandle, LadspaData)>,
    pub deactivate: Option<Deactivate>,
    pub cleanup: Option<Cleanup>,
}

// The descriptor is immutable static metadata; its pointers refer to static arrays/strings.
unsafe impl Sync for Descriptor {}

struct PluginSpec {
    label: &'static std::ffi::CStr,
    factory: fn() -> BackendFactory,
    port_count: c_ulong,
    port_descriptors: *const c_int,
    port_names: *const *const c_char,
    port_range_hints: *const PortRangeHint,
    backend_control_count: usize,
    control_defaults: &'static [f32],
    bypass_control: Option<usize>,
}

// Specs are static; descriptor-array pointers refer to static NUL-terminated metadata.
unsafe impl Sync for PluginSpec {}

struct Instance {
    engine: Engine,
    ports: Vec<*mut LadspaData>,
    backend_control_count: usize,
    control_defaults: &'static [f32],
    bypass_control: Option<usize>,
}

const PORT_DESCRIPTORS: [c_int; 9] = [9, 10, 5, 6, 5, 5, 5, 5, 5];
const PORT_NAMES: [*const c_char; 9] = [
    c"Audio In".as_ptr(),
    c"Audio Out".as_ptr(),
    c"Latency (ms)".as_ptr(),
    c"latency".as_ptr(),
    c"Attenuation Limit (dB)".as_ptr(),
    c"Min processing threshold (dB)".as_ptr(),
    c"Max ERB processing threshold (dB)".as_ptr(),
    c"Max DF processing threshold (dB)".as_ptr(),
    c"Post Filter Beta".as_ptr(),
];
const PORT_HINTS: [PortRangeHint; 9] = [
    PortRangeHint {
        hint_descriptor: 0,
        lower_bound: 0.0,
        upper_bound: 0.0,
    },
    PortRangeHint {
        hint_descriptor: 0,
        lower_bound: 0.0,
        upper_bound: 0.0,
    },
    PortRangeHint {
        hint_descriptor: 131, // BOUNDED_BELOW | BOUNDED_ABOVE | DEFAULT_LOW
        lower_bound: 10.0,
        upper_bound: 200.0,
    },
    PortRangeHint {
        hint_descriptor: 0,
        lower_bound: 0.0,
        upper_bound: 0.0,
    },
    PortRangeHint {
        hint_descriptor: 323, // BOUNDED_BELOW | BOUNDED_ABOVE | DEFAULT_MAXIMUM
        lower_bound: 0.0,
        upper_bound: 100.0,
    },
    PortRangeHint {
        hint_descriptor: 67, // BOUNDED_BELOW | BOUNDED_ABOVE | DEFAULT_MINIMUM
        lower_bound: -15.0,
        upper_bound: 35.0,
    },
    PortRangeHint {
        hint_descriptor: 323,
        lower_bound: -15.0,
        upper_bound: 35.0,
    },
    PortRangeHint {
        hint_descriptor: 323,
        lower_bound: -15.0,
        upper_bound: 35.0,
    },
    PortRangeHint {
        hint_descriptor: 67,
        lower_bound: 0.0,
        upper_bound: 0.05,
    },
];

const fn descriptor(
    spec: &'static PluginSpec,
    unique_id: c_ulong,
    name: &'static std::ffi::CStr,
) -> Descriptor {
    Descriptor {
        unique_id,
        label: spec.label.as_ptr(),
        properties: 0,
        name: name.as_ptr(),
        maker: c"ClearVoice".as_ptr(),
        copyright: c"None".as_ptr(),
        port_count: spec.port_count,
        port_descriptors: spec.port_descriptors,
        port_names: spec.port_names,
        port_range_hints: spec.port_range_hints,
        implementation_data: (spec as *const PluginSpec).cast_mut().cast(),
        instantiate: Some(instantiate),
        connect_port: Some(connect_port),
        activate: Some(activate),
        run: Some(run),
        run_adding: None,
        set_run_adding_gain: None,
        deactivate: Some(deactivate),
        cleanup: Some(cleanup),
    }
}

const DFN_CONTROL_DEFAULTS: [f32; 5] = [100.0, -15.0, 35.0, 35.0, 0.0];

static DFN_SPEC: PluginSpec = PluginSpec {
    label: c"clearvoice_dfn3_ll_mono",
    factory: crate::backend::dfn::DfnBackend::factory,
    port_count: 9,
    port_descriptors: PORT_DESCRIPTORS.as_ptr(),
    port_names: PORT_NAMES.as_ptr(),
    port_range_hints: PORT_HINTS.as_ptr(),
    backend_control_count: 5,
    control_defaults: &DFN_CONTROL_DEFAULTS,
    bypass_control: Some(0),
};

// Fixed local LADSPA ID 0xC1EA03 (below LADSPA hosts' 0x1000000 limit).
static DFN_DESCRIPTOR: Descriptor = descriptor(
    &DFN_SPEC,
    0x00C1_EA03,
    c"ClearVoice DeepFilterNet3-LL (constant latency)",
);

static FASTENHANCER_B_SPEC: PluginSpec = PluginSpec {
    label: c"clearvoice_fastenhancer_b_mono",
    factory: crate::backend::fastenhancer::FastEnhancer::factory_b,
    port_count: 4,
    port_descriptors: PORT_DESCRIPTORS.as_ptr(),
    port_names: PORT_NAMES.as_ptr(),
    port_range_hints: PORT_HINTS.as_ptr(),
    backend_control_count: 0,
    control_defaults: &[],
    bypass_control: None,
};

static FASTENHANCER_B_DESCRIPTOR: Descriptor = descriptor(
    &FASTENHANCER_B_SPEC,
    0x00C1_EA04,
    c"ClearVoice FastEnhancer-B (constant latency)",
);

static FASTENHANCER_S_SPEC: PluginSpec = PluginSpec {
    label: c"clearvoice_fastenhancer_s_mono",
    factory: crate::backend::fastenhancer::FastEnhancer::factory_s,
    port_count: 4,
    port_descriptors: PORT_DESCRIPTORS.as_ptr(),
    port_names: PORT_NAMES.as_ptr(),
    port_range_hints: PORT_HINTS.as_ptr(),
    backend_control_count: 0,
    control_defaults: &[],
    bypass_control: None,
};

static FASTENHANCER_S_DESCRIPTOR: Descriptor = descriptor(
    &FASTENHANCER_S_SPEC,
    0x00C1_EA05,
    c"ClearVoice FastEnhancer-S (constant latency)",
);

static FASTENHANCER_M_SPEC: PluginSpec = PluginSpec {
    label: c"clearvoice_fastenhancer_m_mono",
    factory: crate::backend::fastenhancer::FastEnhancer::factory_m,
    port_count: 4,
    port_descriptors: PORT_DESCRIPTORS.as_ptr(),
    port_names: PORT_NAMES.as_ptr(),
    port_range_hints: PORT_HINTS.as_ptr(),
    backend_control_count: 0,
    control_defaults: &[],
    bypass_control: None,
};

static FASTENHANCER_M_DESCRIPTOR: Descriptor = descriptor(
    &FASTENHANCER_M_SPEC,
    0x00C1_EA06,
    c"ClearVoice FastEnhancer-M (constant latency)",
);

static DFN_INT8_SPEC: PluginSpec = PluginSpec {
    label: c"clearvoice_dfn3_ll_int8_mono",
    factory: crate::backend::dfn_ort::DfnOrt::factory,
    port_count: 9,
    port_descriptors: PORT_DESCRIPTORS.as_ptr(),
    port_names: PORT_NAMES.as_ptr(),
    port_range_hints: PORT_HINTS.as_ptr(),
    backend_control_count: 5,
    control_defaults: &DFN_CONTROL_DEFAULTS,
    bypass_control: Some(0),
};

static DFN_INT8_DESCRIPTOR: Descriptor = descriptor(
    &DFN_INT8_SPEC,
    0x00C1_EA07,
    c"ClearVoice DeepFilterNet3-LL int8 (constant latency)",
);

static DESCRIPTORS: [&Descriptor; 5] = [
    &DFN_DESCRIPTOR,
    &FASTENHANCER_B_DESCRIPTOR,
    &FASTENHANCER_S_DESCRIPTOR,
    &FASTENHANCER_M_DESCRIPTOR,
    &DFN_INT8_DESCRIPTOR,
];

/// LADSPA enumeration entry point.
#[unsafe(no_mangle)]
pub extern "C" fn ladspa_descriptor(index: c_ulong) -> *const Descriptor {
    let Ok(index) = usize::try_from(index) else {
        return ptr::null();
    };
    DESCRIPTORS
        .get(index)
        .map(|descriptor| *descriptor as *const Descriptor)
        .unwrap_or(ptr::null())
}

unsafe extern "C" fn instantiate(
    descriptor: *const Descriptor,
    sample_rate: c_ulong,
) -> LadspaHandle {
    if descriptor.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: descriptor is a host-supplied pointer to a static descriptor returned by this library.
    let spec = unsafe {
        (*descriptor)
            .implementation_data
            .cast::<PluginSpec>()
            .as_ref()
    };
    let Some(spec) = spec else {
        return ptr::null_mut();
    };
    if sample_rate != 48_000 {
        // instantiate runs after the node exists; ClearVoice falls back on this line.
        eprintln!(
            "clearvoice-ladspa fatal label={} reason=unsupported sample rate {sample_rate}",
            spec.label.to_string_lossy()
        );
        return ptr::null_mut();
    }
    if spec.backend_control_count > BACKEND_CONTROL_SLOTS
        || spec.control_defaults.len() != spec.backend_control_count
        || spec.port_count != 4 + spec.backend_control_count as c_ulong
    {
        return ptr::null_mut();
    }
    let label = spec.label.to_str().unwrap_or("invalid-label");
    let engine = match Engine::spawn(label, (spec.factory)()) {
        Ok(engine) => engine,
        Err(_) => return ptr::null_mut(),
    };
    Box::into_raw(Box::new(Instance {
        engine,
        ports: vec![ptr::null_mut(); spec.port_count as usize],
        backend_control_count: spec.backend_control_count,
        control_defaults: spec.control_defaults,
        bypass_control: spec.bypass_control,
    }))
    .cast()
}

unsafe extern "C" fn connect_port(handle: LadspaHandle, port: c_ulong, data: *mut LadspaData) {
    if handle.is_null() {
        return;
    }
    // SAFETY: a non-null LADSPA handle was allocated by `instantiate` and remains host-owned.
    let instance = unsafe { &mut *handle.cast::<Instance>() };
    if let Some(slot) = usize::try_from(port)
        .ok()
        .and_then(|port| instance.ports.get_mut(port))
    {
        *slot = data;
    }
}

unsafe extern "C" fn activate(handle: LadspaHandle) {
    if handle.is_null() {
        return;
    }
    // SAFETY: a non-null LADSPA handle was allocated by `instantiate` and remains host-owned.
    let instance = unsafe { &mut *handle.cast::<Instance>() };
    let control = instance.ports.get(2).copied().unwrap_or(ptr::null_mut());
    let latency_ms = if control.is_null() {
        35.0
    } else {
        // SAFETY: control ports point to one LADSPA_Data supplied by the host.
        unsafe { control.read() }
    };
    let latency = instance.engine.activate(latency_ms) as f32;
    if let Some(port) = instance
        .ports
        .get(3)
        .copied()
        .filter(|port| !port.is_null())
    {
        // SAFETY: output-control ports point to one writable LADSPA_Data supplied by the host.
        unsafe { port.write(latency) };
    }
}

unsafe extern "C" fn run(handle: LadspaHandle, sample_count: c_ulong) {
    if handle.is_null() {
        return;
    }
    // SAFETY: a non-null LADSPA handle was allocated by `instantiate` and remains host-owned.
    let instance = unsafe { &mut *handle.cast::<Instance>() };
    let input = instance.ports.first().copied().unwrap_or(ptr::null_mut());
    let output = instance.ports.get(1).copied().unwrap_or(ptr::null_mut());
    for index in 0..instance.backend_control_count {
        let control = instance
            .ports
            .get(4 + index)
            .copied()
            .unwrap_or(ptr::null_mut());
        let value = if control.is_null() {
            instance.control_defaults.get(index).copied().unwrap_or(0.0)
        } else {
            // SAFETY: backend control ports point to one LADSPA_Data supplied by the host.
            unsafe { control.read() }
        };
        instance.engine.set_backend_control(index, value);
        if instance.bypass_control == Some(index) {
            instance
                .engine
                .set_bypass(crate::backend::dfn::DfnBackend::attenuation_bypasses(value));
        }
    }
    let Ok(count) = usize::try_from(sample_count) else {
        return;
    };
    // SAFETY: null ports are handled as silence/discard by run_raw; non-null ports are host buffers.
    unsafe { instance.engine.run_raw(input, output, count) };
}

unsafe extern "C" fn deactivate(handle: LadspaHandle) {
    if handle.is_null() {
        return;
    }
    // SAFETY: a non-null LADSPA handle was allocated by `instantiate` and remains host-owned.
    let instance = unsafe { &mut *handle.cast::<Instance>() };
    instance.engine.deactivate();
}

unsafe extern "C" fn cleanup(handle: LadspaHandle) {
    if !handle.is_null() {
        // SAFETY: this consumes the unique Box created by `instantiate`.
        drop(unsafe { Box::from_raw(handle.cast::<Instance>()) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Backend, TEST_LOCK};
    use std::sync::OnceLock;

    static CONTROL_TX: OnceLock<std::sync::mpsc::Sender<(usize, u32)>> = OnceLock::new();

    struct AbiBackend;

    impl Backend for AbiBackend {
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
            output.copy_from_slice(input);
            Ok(())
        }
        fn reset(&mut self) {}
    }

    fn test_factory() -> BackendFactory {
        Box::new(|| Ok(Box::new(AbiBackend)))
    }

    static TEST_SPEC: PluginSpec = PluginSpec {
        label: c"clearvoice_test_fake",
        factory: test_factory,
        port_count: 4,
        port_descriptors: PORT_DESCRIPTORS.as_ptr(),
        port_names: PORT_NAMES.as_ptr(),
        port_range_hints: PORT_HINTS.as_ptr(),
        backend_control_count: 0,
        control_defaults: &[],
        bypass_control: None,
    };
    static TEST_DESCRIPTOR: Descriptor =
        descriptor(&TEST_SPEC, 0x434c_5654, c"ClearVoice Test Fake");

    const CONTROL_PORT_DESCRIPTORS: [c_int; 5] = [9, 10, 5, 6, 5];
    const CONTROL_PORT_NAMES: [*const c_char; 5] = [
        c"Audio In".as_ptr(),
        c"Audio Out".as_ptr(),
        c"Latency (ms)".as_ptr(),
        c"latency".as_ptr(),
        c"Fake Control".as_ptr(),
    ];
    const CONTROL_PORT_HINTS: [PortRangeHint; 5] = [
        PORT_HINTS[0],
        PORT_HINTS[1],
        PORT_HINTS[2],
        PORT_HINTS[3],
        PortRangeHint {
            hint_descriptor: 0,
            lower_bound: 0.0,
            upper_bound: 0.0,
        },
    ];
    const CONTROL_DEFAULTS: [f32; 1] = [0.0];

    fn control_factory() -> BackendFactory {
        Box::new(|| Ok(Box::new(ControlProbeBackend)))
    }

    static CONTROL_SPEC: PluginSpec = PluginSpec {
        label: c"clearvoice_test_control",
        factory: control_factory,
        port_count: 5,
        port_descriptors: CONTROL_PORT_DESCRIPTORS.as_ptr(),
        port_names: CONTROL_PORT_NAMES.as_ptr(),
        port_range_hints: CONTROL_PORT_HINTS.as_ptr(),
        backend_control_count: 1,
        control_defaults: &CONTROL_DEFAULTS,
        bypass_control: Some(0),
    };
    static CONTROL_DESCRIPTOR: Descriptor =
        descriptor(&CONTROL_SPEC, 0x434c_5655, c"ClearVoice Test Control");

    struct ControlProbeBackend;

    impl Backend for ControlProbeBackend {
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
            if let Some(sender) = CONTROL_TX.get() {
                let _ = sender.send((index, value.to_bits()));
            }
        }
    }

    #[test]
    fn exported_table_contains_dfn_and_abi_descriptor_matches_ports() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let exported_ptr = ladspa_descriptor(0);
        assert!(!exported_ptr.is_null());
        let exported = unsafe { &*exported_ptr };
        assert!(ladspa_descriptor(5).is_null());
        assert!(ladspa_descriptor(c_ulong::MAX).is_null());
        assert_eq!(exported.unique_id, 0x00C1_EA03);
        assert_eq!(exported.port_count, 9);
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(exported.label) }.to_bytes(),
            b"clearvoice_dfn3_ll_mono"
        );
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(exported.name) }.to_bytes(),
            b"ClearVoice DeepFilterNet3-LL (constant latency)"
        );
        let descriptors = unsafe { std::slice::from_raw_parts(exported.port_descriptors, 9) };
        assert_eq!(descriptors, [9, 10, 5, 6, 5, 5, 5, 5, 5]);
        let exported_names = unsafe { std::slice::from_raw_parts(exported.port_names, 9) };
        let exported_names: Vec<&[u8]> = exported_names
            .iter()
            .map(|name| unsafe { std::ffi::CStr::from_ptr(*name) }.to_bytes())
            .collect();
        assert_eq!(
            exported_names,
            [
                b"Audio In".as_slice(),
                b"Audio Out",
                b"Latency (ms)",
                b"latency",
                b"Attenuation Limit (dB)",
                b"Min processing threshold (dB)",
                b"Max ERB processing threshold (dB)",
                b"Max DF processing threshold (dB)",
                b"Post Filter Beta",
            ]
        );
        let exported_hints = unsafe { std::slice::from_raw_parts(exported.port_range_hints, 9) };
        assert_eq!(exported_hints[2].hint_descriptor, 131);
        assert_eq!(exported_hints[4].hint_descriptor, 323);
        assert_eq!(exported_hints[5].hint_descriptor, 67);
        assert_eq!(exported_hints[6].hint_descriptor, 323);
        assert_eq!(exported_hints[7].hint_descriptor, 323);
        assert_eq!(exported_hints[8].hint_descriptor, 67);
        for (index, id, expected_label, name) in [
            (
                1usize,
                0x00C1_EA04,
                b"clearvoice_fastenhancer_b_mono".as_slice(),
                b"ClearVoice FastEnhancer-B (constant latency)".as_slice(),
            ),
            (
                2usize,
                0x00C1_EA05,
                b"clearvoice_fastenhancer_s_mono".as_slice(),
                b"ClearVoice FastEnhancer-S (constant latency)".as_slice(),
            ),
            (
                3usize,
                0x00C1_EA06,
                b"clearvoice_fastenhancer_m_mono".as_slice(),
                b"ClearVoice FastEnhancer-M (constant latency)".as_slice(),
            ),
        ] {
            let descriptor_ptr = ladspa_descriptor(index as c_ulong);
            assert!(!descriptor_ptr.is_null());
            let descriptor = unsafe { &*descriptor_ptr };
            assert_eq!(descriptor.unique_id, id);
            assert_eq!(descriptor.port_count, 4);
            assert_eq!(
                unsafe { std::ffi::CStr::from_ptr(descriptor.label) }.to_bytes(),
                expected_label
            );
            assert_eq!(
                unsafe { std::ffi::CStr::from_ptr(descriptor.name) }.to_bytes(),
                name
            );
            assert_eq!(
                unsafe { std::slice::from_raw_parts(descriptor.port_descriptors, 4) },
                [9, 10, 5, 6]
            );
        }
        let int8_ptr = ladspa_descriptor(4);
        assert!(!int8_ptr.is_null());
        let int8 = unsafe { &*int8_ptr };
        assert_eq!(int8.unique_id, 0x00C1_EA07);
        assert_eq!(int8.port_count, 9);
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(int8.label) }.to_bytes(),
            b"clearvoice_dfn3_ll_int8_mono"
        );
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(int8.name) }.to_bytes(),
            b"ClearVoice DeepFilterNet3-LL int8 (constant latency)"
        );
        assert_eq!(
            unsafe { std::slice::from_raw_parts(int8.port_descriptors, 9) },
            [9, 10, 5, 6, 5, 5, 5, 5, 5]
        );
        assert_eq!(
            unsafe { std::slice::from_raw_parts(int8.port_names, 9) },
            unsafe { std::slice::from_raw_parts(exported.port_names, 9) }
        );
        assert_eq!(TEST_DESCRIPTOR.port_count, 4);
        assert_eq!(TEST_DESCRIPTOR.properties, 0);
        assert!(TEST_DESCRIPTOR.run_adding.is_none());
        assert!(TEST_DESCRIPTOR.set_run_adding_gain.is_none());
        assert_eq!(
            unsafe { std::ffi::CStr::from_ptr(TEST_DESCRIPTOR.label) }.to_bytes(),
            b"clearvoice_test_fake"
        );
        let names = unsafe { std::slice::from_raw_parts(TEST_DESCRIPTOR.port_names, 4) };
        let names: Vec<&[u8]> = names
            .iter()
            .map(|name| unsafe { std::ffi::CStr::from_ptr(*name) }.to_bytes())
            .collect();
        assert_eq!(
            names,
            [
                b"Audio In".as_slice(),
                b"Audio Out",
                b"Latency (ms)",
                b"latency"
            ]
        );
        let hints = unsafe { std::slice::from_raw_parts(TEST_DESCRIPTOR.port_range_hints, 4) };
        assert_eq!(hints[2].hint_descriptor, 131);
        assert_eq!(hints[2].lower_bound, 10.0);
        assert_eq!(hints[2].upper_bound, 200.0);

        unsafe {
            assert!(instantiate(ptr::null(), 48_000).is_null());
            assert!(instantiate(&TEST_DESCRIPTOR, 44_100).is_null());
            connect_port(ptr::null_mut(), 0, ptr::null_mut());
            activate(ptr::null_mut());
            run(ptr::null_mut(), 0);
            deactivate(ptr::null_mut());
            cleanup(ptr::null_mut());
        }
    }

    #[test]
    fn activate_writes_latched_latency_before_run_and_callbacks_are_null_safe() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut latency_ms = 35.0;
        let mut reported_latency = -1.0;
        let mut audio_in = [1.0; 64];
        let mut audio_out = [0.0; 64];
        let handle = unsafe { instantiate(&TEST_DESCRIPTOR, 48_000) };
        assert!(!handle.is_null());
        unsafe {
            connect_port(handle, 0, audio_in.as_mut_ptr());
            connect_port(handle, 1, audio_out.as_mut_ptr());
            connect_port(handle, 2, &mut latency_ms);
            connect_port(handle, 3, &mut reported_latency);
            connect_port(handle, 99, ptr::null_mut());
            activate(handle);
        }
        assert_eq!(reported_latency, 1680.0);
        unsafe {
            deactivate(handle);
            ptr::write_volatile(&mut latency_ms, 200.0);
            activate(handle);
        }
        assert_eq!(reported_latency, 1680.0);
        unsafe {
            run(handle, 64);
        }
        unsafe {
            connect_port(handle, 0, ptr::null_mut());
            connect_port(handle, 1, ptr::null_mut());
            run(handle, 64);
            deactivate(handle);
            cleanup(handle);
        }
    }

    #[test]
    fn appended_control_ports_reach_the_backend_worker_by_index() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (sender, receiver) = std::sync::mpsc::channel();
        assert!(CONTROL_TX.set(sender).is_ok());
        let handle = unsafe { instantiate(&CONTROL_DESCRIPTOR, 48_000) };
        assert!(!handle.is_null());
        let mut latency_ms = 35.0;
        let mut reported_latency = -1.0;
        let mut control = 12.5;
        let mut input = [0.0; 480];
        let mut output = [0.0; 480];
        unsafe {
            connect_port(handle, 0, input.as_mut_ptr());
            connect_port(handle, 1, output.as_mut_ptr());
            connect_port(handle, 2, &mut latency_ms);
            connect_port(handle, 3, &mut reported_latency);
            connect_port(handle, 4, &mut control);
            activate(handle);
            run(handle, 480);
        }
        assert_eq!(reported_latency, 1680.0);
        let observed = receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("backend control update");
        assert_eq!(observed, (0, 12.5f32.to_bits()));
        unsafe {
            cleanup(handle);
        }
    }
}
