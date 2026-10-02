use std::path::PathBuf;
use std::sync::OnceLock;

use ort::ep::CPU;
use ort::session::{Session, builder::GraphOptimizationLevel};

pub mod dfn;
pub mod fastenhancer;

pub(crate) fn build_ort_session(model: &str, bytes: &[u8]) -> Result<Session, String> {
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
        .clone()?;

    let mut builder =
        Session::builder().map_err(|error| format!("could not create {model} session: {error}"))?;
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
    builder
        .commit_from_memory(bytes)
        .map_err(|error| format!("could not load {model} model: {error}"))
}
