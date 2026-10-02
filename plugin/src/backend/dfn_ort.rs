use std::collections::{HashMap, VecDeque};

use df::{Complex32, DFState, post_filter};
use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::value::{Outlet, Tensor, TensorElementType};

use crate::backend::{build_ort_session, dfn};
use crate::engine::{Backend, BackendFactory};

pub const HOP: usize = 480;
pub const DELAY: usize = 480;
const SAMPLE_RATE: usize = 48_000;
const FFT_SIZE: usize = 960;
const NB_ERB: usize = 32;
const NB_DF: usize = 96;
const DF_ORDER: usize = 5;
const MIN_NB_ERB_FREQS: usize = 2;
const N_FREQS: usize = FFT_SIZE / 2 + 1;
const MAX_GRAPH_INPUTS: usize = 9;
const MODEL_INT8: &str = "DeepFilterNet3-LL int8";
#[cfg(test)]
const MODEL_FP32: &str = "DeepFilterNet3-LL fp32 (test)";

const ENC_INPUTS: &[(&str, &[usize])] = &[
    ("feat_erb", &[1, 1, 1, NB_ERB]),
    ("feat_spec", &[1, 2, 1, NB_DF]),
];
const ERB_INPUTS: &[(&str, &[usize])] = &[
    ("emb", &[1, 1, 512]),
    ("e3", &[1, 64, 1, 8]),
    ("e2", &[1, 64, 1, 8]),
    ("e1", &[1, 64, 1, 16]),
    ("e0", &[1, 64, 1, NB_ERB]),
];
const DF_INPUTS: &[(&str, &[usize])] = &[("emb", &[1, 1, 512]), ("c0", &[1, 64, 1, NB_DF])];
const ENC_OUTPUTS: &[(&str, &[usize])] = &[
    ("e0", &[1, 64, 1, NB_ERB]),
    ("e1", &[1, 64, 1, 16]),
    ("e2", &[1, 64, 1, 8]),
    ("e3", &[1, 64, 1, 8]),
    ("emb", &[1, 1, 512]),
    ("c0", &[1, 64, 1, NB_DF]),
    ("lsnr", &[1, 1, 1]),
];
const ERB_OUTPUTS: &[(&str, &[usize])] = &[("m", &[1, 1, 1, NB_ERB])];
const DF_OUTPUTS: &[(&str, &[usize])] = &[("coefs", &[1, 1, NB_DF, DF_ORDER * 2])];

#[derive(Clone, Copy)]
enum GraphKind {
    Encoder,
    ErbDecoder,
    DfDecoder,
}

impl GraphKind {
    fn name(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::ErbDecoder => "ERB decoder",
            Self::DfDecoder => "DF decoder",
        }
    }

    fn inputs(self) -> &'static [(&'static str, &'static [usize])] {
        match self {
            Self::Encoder => ENC_INPUTS,
            Self::ErbDecoder => ERB_INPUTS,
            Self::DfDecoder => DF_INPUTS,
        }
    }

    fn outputs(self) -> &'static [(&'static str, &'static [usize])] {
        match self {
            Self::Encoder => ENC_OUTPUTS,
            Self::ErbDecoder => ERB_OUTPUTS,
            Self::DfDecoder => DF_OUTPUTS,
        }
    }

    fn state_count(self) -> usize {
        match self {
            Self::Encoder => 7,
            Self::ErbDecoder | Self::DfDecoder => 4,
        }
    }
}

struct StatePair {
    input_index: usize,
    output_name: String,
    shape: Vec<usize>,
}

struct GraphSession {
    session: Session,
    inputs: Vec<Tensor<f32>>,
    input_names: Vec<String>,
    state_pairs: Vec<StatePair>,
    name: String,
    kind: GraphKind,
}

impl GraphSession {
    fn new(kind: GraphKind, model: &str, bytes: &[u8]) -> Result<Self, String> {
        let name = format!("{model} {}", kind.name());
        let session = build_ort_session(&name, bytes)?;
        let mut input_names = Vec::with_capacity(session.inputs().len());
        let mut input_shapes = Vec::with_capacity(session.inputs().len());
        for outlet in session.inputs() {
            let outlet_name = outlet.name().to_owned();
            if input_names.iter().any(|existing| existing == &outlet_name) {
                return Err(format!("{name} has duplicate input {outlet_name}"));
            }
            input_names.push(outlet_name);
            input_shapes.push(static_f32_shape(outlet, &name, "input")?);
        }
        if input_names.is_empty() || input_names.len() > MAX_GRAPH_INPUTS {
            return Err(format!(
                "{name} has {} inputs; expected 1..={MAX_GRAPH_INPUTS}",
                input_names.len()
            ));
        }

        let mut output_shapes = HashMap::with_capacity(session.outputs().len());
        for outlet in session.outputs() {
            let outlet_name = outlet.name().to_owned();
            let shape = static_f32_shape(outlet, &name, "output")?;
            if output_shapes.insert(outlet_name.clone(), shape).is_some() {
                return Err(format!("{name} has duplicate output {outlet_name}"));
            }
        }

        let mut ordinary_input_count = 0;
        let mut state_pairs = Vec::with_capacity(kind.state_count());
        for (input_index, (input_name, shape)) in input_names.iter().zip(&input_shapes).enumerate()
        {
            let Some(state_name) = input_name.strip_suffix("_in") else {
                ordinary_input_count += 1;
                let Some((_, expected_shape)) =
                    kind.inputs().iter().find(|(name, _)| name == input_name)
                else {
                    return Err(format!(
                        "{name} has unexpected non-state input {input_name}"
                    ));
                };
                if shape != *expected_shape {
                    return Err(format!(
                        "{name} input {input_name} has shape {shape:?}, expected {expected_shape:?}"
                    ));
                }
                continue;
            };

            let output_name = format!("{state_name}_out");
            let Some(output_shape) = output_shapes.get(&output_name) else {
                return Err(format!(
                    "{name} state input {input_name} has no {output_name}"
                ));
            };
            if shape != output_shape {
                return Err(format!(
                    "{name} state pair {input_name}/{output_name} has shapes {shape:?} and {output_shape:?}"
                ));
            }
            state_pairs.push(StatePair {
                input_index,
                output_name,
                shape: shape.clone(),
            });
        }
        if ordinary_input_count != kind.inputs().len() {
            return Err(format!(
                "{name} has {ordinary_input_count} ordinary inputs; expected {}",
                kind.inputs().len()
            ));
        }
        if state_pairs.len() != kind.state_count() {
            return Err(format!(
                "{name} has {} state pairs; expected {}",
                state_pairs.len(),
                kind.state_count()
            ));
        }
        for (output_name, expected_shape) in kind.outputs() {
            let Some(shape) = output_shapes.get(*output_name) else {
                return Err(format!("{name} is missing output {output_name}"));
            };
            if shape != *expected_shape {
                return Err(format!(
                    "{name} output {output_name} has shape {shape:?}, expected {expected_shape:?}"
                ));
            }
        }
        for output_name in output_shapes.keys() {
            if output_name.ends_with("_out") {
                if !state_pairs
                    .iter()
                    .any(|pair| pair.output_name == *output_name)
                {
                    return Err(format!("{name} has unpaired state output {output_name}"));
                }
            } else if !kind
                .outputs()
                .iter()
                .any(|(expected, _)| expected == output_name)
                && !(matches!(kind, GraphKind::DfDecoder) && output_name == "302")
            {
                return Err(format!("{name} has unexpected output {output_name}"));
            }
        }

        let inputs = input_shapes
            .iter()
            .map(|shape| tensor(shape, &name))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            session,
            inputs,
            input_names,
            state_pairs,
            name,
            kind,
        })
    }

    fn copy_input(&mut self, input_name: &str, values: &[f32]) -> Result<(), String> {
        let index = self
            .input_names
            .iter()
            .position(|name| name == input_name)
            .ok_or_else(|| format!("{} is missing input {input_name}", self.name))?;
        let input_len = self
            .inputs
            .get(index)
            .ok_or_else(|| format!("{} lost input index for {input_name}", self.name))?
            .extract_tensor()
            .1
            .len();
        if input_len != values.len() {
            return Err(format!(
                "{} input {input_name} has {} values, received {}",
                self.name,
                input_len,
                values.len()
            ));
        }
        let input = self
            .inputs
            .get_mut(index)
            .ok_or_else(|| format!("{} lost input index for {input_name}", self.name))?
            .extract_tensor_mut()
            .1;
        input.copy_from_slice(values);
        Ok(())
    }

    fn copy_output_to_input(
        &mut self,
        input_name: &str,
        outputs: &SessionOutputs<'_>,
        output_name: &str,
        shape: &[usize],
    ) -> Result<(), String> {
        let values = output_data(outputs, &self.name, output_name, shape)?;
        self.copy_input(input_name, values)
    }

    fn run(&mut self) -> Result<SessionOutputs<'_>, String> {
        let input_count = self.inputs.len();
        let input_values: [SessionInputValue<'_>; MAX_GRAPH_INPUTS] =
            std::array::from_fn(|index| {
                self.inputs[index.min(input_count - 1)].upcast_ref().into()
            });
        let outputs = self
            .session
            .run(&input_values[..input_count])
            .map_err(|error| format!("{} inference failed: {error}", self.name))?;
        drop(input_values);

        for pair in &self.state_pairs {
            let values = output_data(&outputs, &self.name, &pair.output_name, &pair.shape)?;
            let Some(input) = self.inputs.get_mut(pair.input_index) else {
                return Err(format!("{} lost state input index", self.name));
            };
            let target = input.extract_tensor_mut().1;
            if target.len() != values.len() {
                return Err(format!(
                    "{} state {} size changed",
                    self.name, pair.output_name
                ));
            }
            target.copy_from_slice(values);
        }
        Ok(outputs)
    }

    fn reset(&mut self) {
        for input in &mut self.inputs {
            input.extract_tensor_mut().1.fill(0.0);
        }
    }
}

pub struct DfnOrt {
    encoder: GraphSession,
    erb_decoder: GraphSession,
    df_decoder: GraphSession,
    df_state: DFState,
    pristine_df_state: DFState,
    spec: Vec<Complex32>,
    rolling_spec_buf_y: VecDeque<Vec<Complex32>>,
    rolling_spec_buf_x: VecDeque<Vec<Complex32>>,
    feat_erb: Vec<f32>,
    feat_cplx: Vec<Complex32>,
    feat_spec: Vec<f32>,
    mask: Vec<f32>,
    coefs: Vec<f32>,
    skip_counter: usize,
    alpha: f32,
    atten_lim: Option<f32>,
    min_db_thresh: f32,
    max_db_erb_thresh: f32,
    max_db_df_thresh: f32,
    post_filter: bool,
    post_filter_beta: f32,
    controls: [f32; 5],
}

impl DfnOrt {
    pub fn new() -> Result<Self, String> {
        Self::from_graph_bytes(
            include_bytes!("../../models/dfn3_ll_int8_enc.onnx"),
            include_bytes!("../../models/dfn3_ll_int8_erb_dec.onnx"),
            include_bytes!("../../models/dfn3_ll_int8_df_dec.onnx"),
            MODEL_INT8,
        )
    }

    pub(crate) fn from_graph_bytes(
        encoder: &[u8],
        erb_decoder: &[u8],
        df_decoder: &[u8],
        model: &str,
    ) -> Result<Self, String> {
        let encoder = GraphSession::new(GraphKind::Encoder, model, encoder)?;
        let erb_decoder = GraphSession::new(GraphKind::ErbDecoder, model, erb_decoder)?;
        let df_decoder = GraphSession::new(GraphKind::DfDecoder, model, df_decoder)?;
        let mut df_state = DFState::new(SAMPLE_RATE, FFT_SIZE, HOP, NB_ERB, MIN_NB_ERB_FREQS);
        df_state.init_norm_states(NB_DF);
        let pristine_df_state = df_state.clone();
        let zero_spectra = || {
            (0..DF_ORDER)
                .map(|_| vec![Complex32::default(); N_FREQS])
                .collect::<VecDeque<_>>()
        };

        Ok(Self {
            encoder,
            erb_decoder,
            df_decoder,
            df_state,
            pristine_df_state,
            spec: vec![Complex32::default(); N_FREQS],
            rolling_spec_buf_y: zero_spectra(),
            rolling_spec_buf_x: zero_spectra(),
            feat_erb: vec![0.0; NB_ERB],
            feat_cplx: vec![Complex32::default(); NB_DF],
            feat_spec: vec![0.0; 2 * NB_DF],
            mask: vec![0.0; NB_ERB],
            coefs: vec![0.0; NB_DF * DF_ORDER * 2],
            skip_counter: 0,
            alpha: calc_norm_alpha(SAMPLE_RATE, HOP, 1.0),
            atten_lim: None,
            min_db_thresh: dfn::CONTROL_DEFAULTS[1],
            max_db_erb_thresh: dfn::CONTROL_DEFAULTS[2],
            max_db_df_thresh: dfn::CONTROL_DEFAULTS[3],
            post_filter: false,
            post_filter_beta: dfn::CONTROL_DEFAULTS[4],
            controls: dfn::CONTROL_DEFAULTS,
        })
    }

    pub fn factory() -> BackendFactory {
        Box::new(|| Ok(Box::new(Self::new()?) as Box<dyn Backend>))
    }

    fn apply_control(&mut self, index: usize, value: f32) {
        let Some(value) = dfn::normalized_control(index, value) else {
            return;
        };
        // The engine performs the aligned dry bypass; don't hand libDF its zero-delay mode.
        if index == 0 && dfn::attenuation_bypasses(value) {
            return;
        }
        if self.controls[index].to_bits() == value.to_bits() {
            return;
        }
        self.controls[index] = value;
        match index {
            0 => {
                let limit = value.abs();
                self.atten_lim = if limit >= 100.0 {
                    None
                } else if limit < 0.01 {
                    Some(1.0)
                } else {
                    Some(10.0f32.powf(-limit / 20.0))
                };
            }
            1 => self.min_db_thresh = value,
            2 => self.max_db_erb_thresh = value,
            3 => self.max_db_df_thresh = value,
            4 => {
                self.post_filter_beta = value;
                self.post_filter = value > 0.0;
            }
            _ => {}
        }
    }

    fn process_frame(&mut self, input: &[f32], output: &mut [f32]) -> Result<f32, String> {
        if input.len() != HOP || output.len() != HOP {
            return Err(format!(
                "expected {HOP}-sample frame, got input={} output={}",
                input.len(),
                output.len()
            ));
        }

        let energy = input.iter().map(|sample| sample.powi(2)).sum::<f32>();
        if energy / (input.len() as f32) < 1e-7 {
            self.skip_counter += 1;
        } else {
            self.skip_counter = 0;
        }
        if self.skip_counter > 5 {
            output.fill(0.0);
            return Ok(-15.0);
        }

        self.df_state.analysis(input, &mut self.spec);
        let mut y_frame = self
            .rolling_spec_buf_y
            .pop_front()
            .ok_or_else(|| "DeepFilterNet enhanced spectrum history is empty".to_owned())?;
        let mut x_frame = self
            .rolling_spec_buf_x
            .pop_front()
            .ok_or_else(|| "DeepFilterNet noisy spectrum history is empty".to_owned())?;
        y_frame.copy_from_slice(&self.spec);
        x_frame.copy_from_slice(&self.spec);
        self.rolling_spec_buf_y.push_back(y_frame);
        self.rolling_spec_buf_x.push_back(x_frame);

        if self.atten_lim == Some(1.0) {
            output.copy_from_slice(input);
            return Ok(35.0);
        }

        self.df_state
            .feat_erb(&self.spec, self.alpha, &mut self.feat_erb);
        self.df_state
            .feat_cplx(&self.spec[..NB_DF], self.alpha, &mut self.feat_cplx);
        for (index, value) in self.feat_cplx.iter().enumerate() {
            self.feat_spec[index] = value.re;
            self.feat_spec[NB_DF + index] = value.im;
        }
        self.encoder.copy_input("feat_erb", &self.feat_erb)?;
        self.encoder.copy_input("feat_spec", &self.feat_spec)?;

        let (min_db_thresh, max_db_erb_thresh, max_db_df_thresh) = (
            self.min_db_thresh,
            self.max_db_erb_thresh,
            self.max_db_df_thresh,
        );
        let encoder_name = self.encoder.kind.name();
        let enc_outputs = self.encoder.run()?;
        let lsnr_values = output_data(&enc_outputs, encoder_name, "lsnr", &[1, 1, 1])?;
        let Some(&lsnr) = lsnr_values.first() else {
            return Err("DeepFilterNet encoder returned an empty lsnr".to_owned());
        };
        let (apply_gains, apply_gain_zeros, apply_df) =
            apply_stages(lsnr, min_db_thresh, max_db_erb_thresh, max_db_df_thresh);

        if apply_gains {
            for (name, shape) in [
                ("emb", &[1, 1, 512][..]),
                ("e3", &[1, 64, 1, 8][..]),
                ("e2", &[1, 64, 1, 8][..]),
                ("e1", &[1, 64, 1, 16][..]),
                ("e0", &[1, 64, 1, NB_ERB][..]),
            ] {
                self.erb_decoder
                    .copy_output_to_input(name, &enc_outputs, name, shape)?;
            }
            let erb_name = self.erb_decoder.kind.name();
            let erb_outputs = self.erb_decoder.run()?;
            let gains = output_data(&erb_outputs, erb_name, "m", &[1, 1, 1, NB_ERB])?;
            self.mask.copy_from_slice(gains);
            self.skip_counter = 0;
        } else if apply_gain_zeros {
            self.mask.fill(0.0);
            self.skip_counter = 0;
        } else {
            self.skip_counter += 1;
        }

        let mut coefs_available = false;
        if apply_df {
            for (name, shape) in [("emb", &[1, 1, 512][..]), ("c0", &[1, 64, 1, NB_DF][..])] {
                self.df_decoder
                    .copy_output_to_input(name, &enc_outputs, name, shape)?;
            }
            let df_name = self.df_decoder.kind.name();
            let df_outputs = self.df_decoder.run()?;
            let values = output_data(&df_outputs, df_name, "coefs", &[1, 1, NB_DF, DF_ORDER * 2])?;
            self.coefs.copy_from_slice(values);
            coefs_available = true;
        }

        if apply_gains || apply_gain_zeros {
            let Some(spectrum) = self.rolling_spec_buf_y.get_mut(DF_ORDER - 1) else {
                return Err("DeepFilterNet enhanced spectrum history is incomplete".to_owned());
            };
            self.df_state.apply_mask(spectrum, &self.mask);
        }
        let Some(enhanced) = self.rolling_spec_buf_y.get(DF_ORDER - 1) else {
            return Err("DeepFilterNet enhanced spectrum history is incomplete".to_owned());
        };
        self.spec.copy_from_slice(enhanced);
        if coefs_available {
            apply_df_filter(&self.rolling_spec_buf_x, &self.coefs, &mut self.spec)?;
        }

        let Some(noisy) = self.rolling_spec_buf_x.get(DF_ORDER - 1) else {
            return Err("DeepFilterNet noisy spectrum history is incomplete".to_owned());
        };
        if apply_gains && self.post_filter {
            post_filter(noisy, &mut self.spec, self.post_filter_beta);
        }
        if let Some(limit) = self.atten_lim {
            let keep = 1.0 - limit;
            for (enhanced, &noisy) in self.spec.iter_mut().zip(noisy) {
                *enhanced *= keep;
                *enhanced += noisy * limit;
            }
        }
        self.df_state.synthesis(&mut self.spec, output);
        Ok(lsnr)
    }

    fn reset_history(&mut self) {
        self.encoder.reset();
        self.erb_decoder.reset();
        self.df_decoder.reset();
        self.df_state = self.pristine_df_state.clone();
        self.spec.fill(Complex32::default());
        for spectrum in self
            .rolling_spec_buf_y
            .iter_mut()
            .chain(self.rolling_spec_buf_x.iter_mut())
        {
            spectrum.fill(Complex32::default());
        }
        self.feat_erb.fill(0.0);
        self.feat_cplx.fill(Complex32::default());
        self.feat_spec.fill(0.0);
        self.mask.fill(0.0);
        self.coefs.fill(0.0);
        self.skip_counter = 0;
    }
}

fn apply_stages(
    lsnr: f32,
    min_db_thresh: f32,
    max_db_erb_thresh: f32,
    max_db_df_thresh: f32,
) -> (bool, bool, bool) {
    if lsnr < min_db_thresh {
        (false, true, false)
    } else if lsnr > max_db_erb_thresh {
        (false, false, false)
    } else if lsnr > max_db_df_thresh {
        (true, false, false)
    } else {
        (true, false, true)
    }
}

impl Backend for DfnOrt {
    fn hop(&self) -> usize {
        HOP
    }

    fn delay(&self) -> usize {
        DELAY
    }

    fn settle_frames(&self) -> usize {
        DELAY.div_ceil(HOP)
    }

    fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        self.process_frame(input, output).map(|_| ())
    }

    fn reset(&mut self) {
        let controls = self.controls;
        self.reset_history();
        self.controls = dfn::CONTROL_DEFAULTS;
        self.atten_lim = None;
        self.min_db_thresh = dfn::CONTROL_DEFAULTS[1];
        self.max_db_erb_thresh = dfn::CONTROL_DEFAULTS[2];
        self.max_db_df_thresh = dfn::CONTROL_DEFAULTS[3];
        self.post_filter = false;
        self.post_filter_beta = dfn::CONTROL_DEFAULTS[4];
        for (index, value) in controls.into_iter().enumerate() {
            self.apply_control(index, value);
        }
    }

    fn set_control(&mut self, index: usize, value: f32) {
        self.apply_control(index, value);
    }
}

fn tensor(shape: &[usize], name: &str) -> Result<Tensor<f32>, String> {
    let size = shape
        .iter()
        .try_fold(1usize, |size, dimension| size.checked_mul(*dimension))
        .ok_or_else(|| format!("{name} tensor size overflows usize"))?;
    Tensor::from_array((shape.to_vec(), vec![0.0f32; size]))
        .map_err(|error| format!("could not allocate {name} input tensor: {error}"))
}

fn static_f32_shape(outlet: &Outlet, name: &str, direction: &str) -> Result<Vec<usize>, String> {
    if outlet.dtype().tensor_type() != Some(TensorElementType::Float32) {
        return Err(format!(
            "{name} {direction} {} must be a float32 tensor",
            outlet.name()
        ));
    }
    let shape = outlet.dtype().tensor_shape().ok_or_else(|| {
        format!(
            "{name} {direction} {} must have a static tensor shape",
            outlet.name()
        )
    })?;
    if shape.is_empty() {
        return Err(format!(
            "{name} {direction} {} has an empty tensor shape",
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
                        "{name} {direction} {} has invalid shape {shape:?}",
                        outlet.name()
                    )
                })
        })
        .collect()
}

fn output_data<'a>(
    outputs: &'a SessionOutputs<'_>,
    model: &str,
    output_name: &str,
    expected_shape: &[usize],
) -> Result<&'a [f32], String> {
    let output = outputs
        .get(output_name)
        .ok_or_else(|| format!("{model} is missing output {output_name}"))?;
    let (shape, values) = output
        .try_extract_tensor::<f32>()
        .map_err(|error| format!("{model} output {output_name} is invalid: {error}"))?;
    if shape.len() != expected_shape.len()
        || shape
            .iter()
            .zip(expected_shape)
            .any(|(actual, expected)| *actual != *expected as i64)
    {
        return Err(format!(
            "{model} output {output_name} has runtime shape {shape:?}, expected {expected_shape:?}"
        ));
    }
    let expected_len = expected_shape.iter().product::<usize>();
    if values.len() != expected_len {
        return Err(format!(
            "{model} output {output_name} has {} values, expected {expected_len}",
            values.len()
        ));
    }
    Ok(values)
}

fn calc_norm_alpha(sr: usize, hop_size: usize, tau: f32) -> f32 {
    let dt = hop_size as f32 / sr as f32;
    let alpha = f32::exp(-dt / tau);
    let mut rounded = 1.0;
    let mut precision = 3;
    while rounded >= 1.0 {
        let scale = 10i32.pow(precision) as f32;
        rounded = (alpha * scale).round() / scale;
        precision += 1;
    }
    rounded
}

fn apply_df_filter(
    spectra: &VecDeque<Vec<Complex32>>,
    coefs: &[f32],
    output: &mut [Complex32],
) -> Result<(), String> {
    if spectra.len() < DF_ORDER || coefs.len() != NB_DF * DF_ORDER * 2 || output.len() < NB_DF {
        return Err("DeepFilterNet DF coefficient or spectrum geometry is invalid".to_owned());
    }
    output[..NB_DF].fill(Complex32::default());
    for (order, spectrum) in spectra.iter().take(DF_ORDER).enumerate() {
        if spectrum.len() != N_FREQS {
            return Err("DeepFilterNet DF spectrum history has invalid geometry".to_owned());
        }
        for (bin, out) in output[..NB_DF].iter_mut().enumerate() {
            let index = (bin * DF_ORDER + order) * 2;
            let coefficient = Complex32::new(coefs[index], coefs[index + 1]);
            *out += spectrum[bin] * coefficient;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::dfn::DfnBackend;
    use crate::engine::TEST_LOCK;

    const PARITY_SECONDS: usize = 20;

    fn fp32_backend() -> DfnOrt {
        let models = [
            "dfn3_ll_fp32_enc.onnx",
            "dfn3_ll_fp32_erb_dec.onnx",
            "dfn3_ll_fp32_df_dec.onnx",
        ]
        .map(|name| {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .join(name);
            std::fs::read(&path).unwrap_or_else(|error| {
                panic!(
                    "required fp32 test graph {} is missing: {error}",
                    path.display()
                )
            })
        });
        DfnOrt::from_graph_bytes(&models[0], &models[1], &models[2], MODEL_FP32)
            .expect("fp32 ORT DeepFilterNet graphs")
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

    fn speech_like_signal() -> Vec<f32> {
        let frames = PARITY_SECONDS * SAMPLE_RATE / HOP;
        let mut signal = Vec::with_capacity(frames * HOP);
        let mut random = 0xd4c3_b2a1_91e7_6f05_u64;
        let mut phase = 0.0f64;
        for sample in 0..frames * HOP {
            let frame = sample / HOP;
            if (800..812).contains(&frame) {
                signal.push(0.0);
                continue;
            }
            let time = sample as f64 / SAMPLE_RATE as f64;
            let f0 = 105.0 + 85.0 * time / PARITY_SECONDS as f64;
            phase += std::f64::consts::TAU * f0 / SAMPLE_RATE as f64;
            let harmonics = (1..=9)
                .map(|harmonic| (phase * harmonic as f64).sin() / harmonic as f64)
                .sum::<f64>();
            let syllable_phase = sample % (SAMPLE_RATE / 4);
            let gate = if syllable_phase < SAMPLE_RATE / 5 {
                (std::f64::consts::PI * syllable_phase as f64 / (SAMPLE_RATE / 5) as f64)
                    .sin()
                    .max(0.0)
            } else {
                0.0
            };
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let noise = (random >> 40) as f64 / ((1_u64 << 24) - 1) as f64 - 0.5;
            let (voice_level, noise_level) = if (1_200..1_600).contains(&frame) {
                (0.018, 0.0003)
            } else {
                (0.075, 0.012)
            };
            signal.push((harmonics * gate * voice_level + noise * noise_level) as f32);
        }
        signal
    }

    fn process_frames(backend: &mut DfnOrt, input: &[f32]) -> Vec<f32> {
        assert!(input.len().is_multiple_of(HOP));
        let mut output = vec![0.0; input.len()];
        let (sources, input_remainder) = input.as_chunks::<HOP>();
        let (targets, output_remainder) = output.as_chunks_mut::<HOP>();
        assert!(input_remainder.is_empty() && output_remainder.is_empty());
        for (source, target) in sources.iter().zip(targets) {
            backend
                .process(source, target)
                .expect("DeepFilterNet ORT frame");
        }
        output
    }

    fn correlation(left: &[f32], right: &[f32]) -> f64 {
        let dot = left
            .iter()
            .zip(right)
            .map(|(x, y)| *x as f64 * *y as f64)
            .sum::<f64>();
        let power = |samples: &[f32]| {
            samples
                .iter()
                .map(|sample| (*sample as f64).powi(2))
                .sum::<f64>()
        };
        dot / (power(left) * power(right)).sqrt().max(f64::MIN_POSITIVE)
    }

    fn reference_tract() -> DfnBackend {
        DfnBackend::new().expect("tract DeepFilterNet3-LL")
    }

    #[test]
    fn fp32_ort_matches_tract_on_speech_silence_and_clean_frames() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let signal = speech_like_signal();
        assert!(signal.len() >= 20 * SAMPLE_RATE);
        assert!(
            signal[800 * HOP..812 * HOP]
                .iter()
                .all(|sample| *sample == 0.0)
        );
        let mut ort_backend = fp32_backend();
        let mut tract = reference_tract();
        let mut tract_output = [0.0f32; HOP];
        let mut ort_output = [0.0f32; HOP];
        let mut reference_power = 0.0f64;
        let mut error_power = 0.0f64;
        let mut control_reference_power = 0.0f64;
        let mut control_error_power = 0.0f64;
        let mut silence_reference_power = 0.0f64;
        let mut silence_error_power = 0.0f64;
        let mut silence_reference_max = 0.0f32;
        let mut silence_max_abs_diff = 0.0f32;
        let mut resume_reference_power = 0.0f64;
        let mut resume_error_power = 0.0f64;
        let mut resume_reference_max = 0.0f32;
        let mut resume_max_abs_diff = 0.0f32;
        let mut max_abs_diff = 0.0f32;
        let mut decisions = [0usize; 4];
        let mut compared = 0usize;
        let mut thresholds = (-15.0f32, 35.0f32, 35.0f32);

        let (signal_frames, remainder) = signal.as_chunks::<HOP>();
        assert!(remainder.is_empty());
        for (frame, source) in signal_frames.iter().enumerate() {
            if frame == 500 {
                ort_backend.set_control(0, 12.0);
                ort_backend.set_control(4, 0.02);
                tract.set_control(0, 12.0);
                tract.set_control(4, 0.02);
            } else if frame == 700 {
                ort_backend.set_control(0, 100.0);
                ort_backend.set_control(4, 0.0);
                tract.set_control(0, 100.0);
                tract.set_control(4, 0.0);
            } else if frame == 800 {
                ort_backend.set_control(2, -15.0);
                tract.set_control(2, -15.0);
                thresholds.1 = -15.0;
            } else if frame == 812 {
                ort_backend.set_control(2, 35.0);
                tract.set_control(2, 35.0);
                thresholds.1 = 35.0;
            } else if frame == 1_200 {
                ort_backend.set_control(2, -15.0);
                ort_backend.set_control(3, -15.0);
                tract.set_control(2, -15.0);
                tract.set_control(3, -15.0);
                thresholds.1 = -15.0;
                thresholds.2 = -15.0;
            } else if frame == 1_250 {
                ort_backend.set_control(1, 35.0);
                tract.set_control(1, 35.0);
                thresholds.0 = 35.0;
            } else if frame == 1_300 {
                ort_backend.set_control(1, -15.0);
                ort_backend.set_control(2, 35.0);
                tract.set_control(1, -15.0);
                tract.set_control(2, 35.0);
                thresholds.0 = -15.0;
                thresholds.1 = 35.0;
            } else if frame == 1_400 {
                ort_backend.set_control(3, 35.0);
                tract.set_control(3, 35.0);
                thresholds.2 = 35.0;
            }
            let ort_lsnr = ort_backend
                .process_frame(source, &mut ort_output)
                .expect("fp32 ORT frame");
            tract
                .process(source, &mut tract_output)
                .expect("tract frame");
            let ort_decision = apply_stages(ort_lsnr, thresholds.0, thresholds.1, thresholds.2);
            let decision_index = match ort_decision {
                (false, true, false) => 0,
                (false, false, false) => 1,
                (true, false, false) => 2,
                (true, false, true) => 3,
                _ => unreachable!(),
            };
            decisions[decision_index] += 1;
            if (800..812).contains(&frame) {
                let input_rms = (source
                    .iter()
                    .map(|sample| (*sample as f64).powi(2))
                    .sum::<f64>()
                    / HOP as f64)
                    .sqrt();
                assert!(input_rms < 1e-7, "silence frame {frame} RMS={input_rms}");
            }
            if (803..812).contains(&frame) {
                assert!(
                    ort_backend.skip_counter > 5,
                    "fp32 ORT skip counter did not exceed five at frame {frame}"
                );
                assert!(
                    ort_output.iter().all(|sample| *sample == 0.0),
                    "fp32 ORT silence skip did not zero frame {frame}"
                );
                assert!(
                    tract_output.iter().all(|sample| sample.abs() <= 1e-7),
                    "tract silence skip did not zero frame {frame}"
                );
            }
            if frame >= 5 {
                for (&actual, &reference) in ort_output.iter().zip(&tract_output) {
                    let diff = actual - reference;
                    max_abs_diff = max_abs_diff.max(diff.abs());
                    reference_power += (reference as f64).powi(2);
                    error_power += (diff as f64).powi(2);
                    if (500..700).contains(&frame) {
                        control_reference_power += (reference as f64).powi(2);
                        control_error_power += (diff as f64).powi(2);
                    }
                    if (800..812).contains(&frame) {
                        silence_reference_power += (reference as f64).powi(2);
                        silence_error_power += (diff as f64).powi(2);
                        silence_reference_max = silence_reference_max.max(reference.abs());
                        silence_max_abs_diff = silence_max_abs_diff.max(diff.abs());
                    }
                    if (812..827).contains(&frame) {
                        resume_reference_power += (reference as f64).powi(2);
                        resume_error_power += (diff as f64).powi(2);
                        resume_reference_max = resume_reference_max.max(reference.abs());
                        resume_max_abs_diff = resume_max_abs_diff.max(diff.abs());
                    }
                }
                compared += 1;
            }
        }
        let snr_db = 10.0 * (reference_power / error_power.max(f64::MIN_POSITIVE)).log10();
        eprintln!(
            "fp32 ORT vs tract: SNR={snr_db:.2} dB max_abs_diff={max_abs_diff:.8e} frames={} decisions={decisions:?}",
            signal.len().div_ceil(HOP),
        );
        let control_snr_db =
            10.0 * (control_reference_power / control_error_power.max(f64::MIN_POSITIVE)).log10();
        eprintln!("attenuation/post-filter control segment SNR={control_snr_db:.2} dB");
        assert!(
            control_reference_power > 0.0,
            "control segment has no reference power"
        );
        assert!(
            control_snr_db >= 60.0,
            "controlled-segment parity SNR {control_snr_db:.2} dB < 60 dB"
        );
        if silence_reference_max <= 1e-5 {
            assert!(
                silence_max_abs_diff <= 1e-5,
                "silence-window max diff {silence_max_abs_diff:.8e} > 1e-5"
            );
        } else {
            let silence_snr_db = 10.0
                * (silence_reference_power / silence_error_power.max(f64::MIN_POSITIVE)).log10();
            eprintln!("silence-window parity SNR={silence_snr_db:.2} dB");
            assert!(
                silence_snr_db >= 40.0,
                "silence-window SNR {silence_snr_db:.2} dB < 40 dB"
            );
        }
        if resume_reference_max <= 1e-5 {
            assert!(
                resume_max_abs_diff <= 1e-5,
                "resume-window max diff {resume_max_abs_diff:.8e} > 1e-5"
            );
        } else {
            let resume_snr_db =
                10.0 * (resume_reference_power / resume_error_power.max(f64::MIN_POSITIVE)).log10();
            eprintln!("15-frame resume-window parity SNR={resume_snr_db:.2} dB");
            assert!(
                resume_snr_db >= 40.0,
                "resume-window SNR {resume_snr_db:.2} dB < 40 dB"
            );
        }
        assert!(compared >= 20 * SAMPLE_RATE / HOP - 5);
        assert!(snr_db >= 40.0, "fp32 parity SNR {snr_db:.2} dB < 40 dB");
        assert!(decisions[0] > 0, "no zero-mask frames were tested");
        assert!(decisions[1] > 0, "no clean-frame stage skips were tested");
        assert!(decisions[2] > 0, "no ERB-only frames were tested");
        assert!(decisions[3] > 0, "no full DF frames were tested");
    }

    #[test]
    fn int8_vs_fp32_ort_reports_end_to_end_snr_and_finite_audio() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let signal = speech_like_signal();
        let mut int8 = DfnOrt::new().expect("int8 ORT DeepFilterNet3-LL");
        let mut fp32 = fp32_backend();
        let int8_output = process_frames(&mut int8, &signal);
        let fp32_output = process_frames(&mut fp32, &signal);
        assert!(int8_output.iter().all(|sample| sample.is_finite()));
        assert!(fp32_output.iter().all(|sample| sample.is_finite()));
        let reference_power = fp32_output
            .iter()
            .map(|sample| (*sample as f64).powi(2))
            .sum::<f64>();
        let error_power = int8_output
            .iter()
            .zip(&fp32_output)
            .map(|(actual, reference)| ((*actual - *reference) as f64).powi(2))
            .sum::<f64>();
        assert!(
            reference_power > 0.0,
            "fp32 ORT reference output has zero power"
        );
        let snr_db = 10.0 * (reference_power / error_power.max(f64::MIN_POSITIVE)).log10();
        eprintln!("int8 vs fp32 ORT output SNR={snr_db:.2} dB");
        assert!(snr_db >= 35.0, "int8 output SNR {snr_db:.2} dB < 35 dB");
    }

    #[test]
    fn delay_is_480_by_correlation_and_reset_matches_fresh_with_derived_settling() {
        let _guard = TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut backend = DfnOrt::new().expect("int8 ORT DeepFilterNet3-LL");
        backend.set_control(0, 1.0);
        let input = broadband(40);
        let output = process_frames(&mut backend, &input);
        let start = 3 * HOP;
        let mut best = (f64::NEG_INFINITY, 0usize);
        for lag in 0..=2 * HOP {
            let dot = (start..input.len())
                .map(|index| input[index - lag] as f64 * output[index] as f64)
                .sum::<f64>();
            let input_power = (start..input.len())
                .map(|index| (input[index - lag] as f64).powi(2))
                .sum::<f64>();
            let output_power = (start..input.len())
                .map(|index| (output[index] as f64).powi(2))
                .sum::<f64>();
            let value = dot / (input_power * output_power).sqrt();
            if value > best.0 {
                best = (value, lag);
            }
        }
        assert_eq!(best.1, DELAY, "best normalized correlation {}", best.0);
        assert!(best.0 > 0.8, "delay correlation too weak: {}", best.0);

        let mut backend = DfnOrt::new().expect("int8 ORT reset backend");
        let mut fresh = DfnOrt::new().expect("int8 ORT fresh backend");
        for candidate in [&mut backend, &mut fresh] {
            candidate.set_control(0, 1.0);
            candidate.set_control(4, 0.02);
        }
        let prior = broadband(8);
        let stream = broadband(10);
        let _ = process_frames(&mut backend, &prior);
        backend.reset();
        let output = process_frames(&mut backend, &stream[HOP..]);
        let oracle = process_frames(&mut fresh, &stream[HOP..]);
        assert_eq!(
            output, oracle,
            "reset must match a fresh backend bit-for-bit"
        );
        assert_eq!(backend.settle_frames(), DELAY.div_ceil(HOP));
        assert_eq!(backend.settle_frames(), 1);
        let first_valid = (0..8)
            .find(|&frame| {
                correlation(
                    &output[frame * HOP..(frame + 1) * HOP],
                    &stream[frame * HOP..(frame + 1) * HOP],
                ) > 0.8
            })
            .expect("first aligned post-reset frame");
        assert_eq!(first_valid, backend.settle_frames());
    }
}
