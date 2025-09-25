mod array;
mod ffi;
mod python_vm;
mod vm;
//mod python_bindings;
//mod vm;

pub use crate::{
    array::{Buffer, DType, Order, Shape, Tensor, TensorType},
    ffi::{CallCommandArg, ExternalArrayFunc, NumbaRuntime},
    vm::{
        DynInstruction, Error, ExecutionContext, FunctionBuilder, GlobalValueId, Instruction,
        Module, ModuleBuilder, NodeId, RayonExecutionContext, RayonValueStore, Region,
        RegionBuilder, RegionId, Value, ValueId, ValueStore, ValueType, Variable,
    },
};

use pyo3::prelude::*;

/// Python module for PyTensor-RT
#[pymodule]
fn _lib(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add graph building classes
    python_vm::create_python_module(py, m)?;

    Ok(())
}
