use std::{cell::RefCell, rc::Rc, sync::Arc};

use anyhow::{Context, Result};
use numpy::{
    PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::{prelude::*, types::PyType};

use crate::{
    vm::VariableType, Alloc, Array, Buffer, Command, CommandWithVariables, DType,
    ExternalArrayFunc, ExternalCall, NumbaRuntime, Order, Variable, VM,
};

// Python wrapper for DType
#[pyclass(name = "DType")]
#[derive(Clone)]
pub struct PyDType {
    inner: DType,
}

#[pymethods]
impl PyDType {
    #[staticmethod]
    fn f64() -> Self {
        PyDType { inner: DType::F64 }
    }

    #[staticmethod]
    fn f32() -> Self {
        PyDType { inner: DType::F32 }
    }

    #[staticmethod]
    fn i64() -> Self {
        PyDType { inner: DType::I64 }
    }

    #[staticmethod]
    fn i32() -> Self {
        PyDType { inner: DType::I32 }
    }

    #[staticmethod]
    fn u64() -> Self {
        PyDType { inner: DType::U64 }
    }

    #[staticmethod]
    fn u32() -> Self {
        PyDType { inner: DType::U32 }
    }

    #[staticmethod]
    fn i16() -> Self {
        PyDType { inner: DType::I16 }
    }

    #[staticmethod]
    fn u16() -> Self {
        PyDType { inner: DType::U16 }
    }

    #[staticmethod]
    fn i8() -> Self {
        PyDType { inner: DType::I8 }
    }

    #[staticmethod]
    fn u8() -> Self {
        PyDType { inner: DType::U8 }
    }

    #[staticmethod]
    fn bool() -> Self {
        PyDType { inner: DType::Bool }
    }

    #[staticmethod]
    fn from_string(s: &str) -> PyResult<Self> {
        DType::from_string(s)
            .map(|inner| PyDType { inner })
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unknown dtype: {}", s))
            })
    }

    fn __str__(&self) -> &'static str {
        match self.inner {
            DType::F64 => "f64",
            DType::F32 => "f32",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::U64 => "u64",
            DType::U32 => "u32",
            DType::I16 => "i16",
            DType::U16 => "u16",
            DType::I8 => "i8",
            DType::U8 => "u8",
            DType::Bool => "bool",
        }
    }

    fn size_bytes(&self) -> usize {
        self.inner.size_bytes()
    }
}

// Python wrapper for Alloc
#[pyclass(name = "Alloc")]
#[derive(Clone)]
pub struct PyAlloc {
    shape: Vec<usize>,
    dtype: PyDType,
    order: Order,
}

#[pymethods]
impl PyAlloc {
    #[new]
    fn new(shape: Vec<usize>, dtype: PyDType, order: Option<String>) -> PyResult<Self> {
        let order = order.unwrap_or_else(|| "C".to_string());

        // Validate order
        let order = match order.as_str() {
            "C" | "row_major" => Order::c(),
            "F" | "column_major" => Order::f(),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown order: {}",
                    order
                )))
            }
        };

        Ok(PyAlloc {
            shape,
            dtype,
            order,
        })
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[getter]
    fn dtype(&self) -> PyDType {
        self.dtype.clone()
    }

    #[getter]
    fn order(&self) -> String {
        self.order.to_string()
    }
}

impl PyAlloc {
    pub fn to_alloc(&self) -> Alloc {
        Alloc::new(self.shape.clone().into(), self.dtype.inner, self.order)
    }
}

// Python wrapper for CallFuncPtr
#[pyclass(name = "ExternalCall")]
#[derive(Clone)]
pub struct PyExternalCall {
    func: ExternalCall,
}

#[pymethods]
impl PyExternalCall {
    #[new]
    /// # Safety
    /// The function pointer must be valid and have the correct signature.
    fn new(func_ptr: usize, keep_alive: Option<Py<PyAny>>) -> Self {
        // SAFETY: We trust the user to provide a valid function pointer
        // that lives at least as long as keep_alive.
        let func_ptr = unsafe { std::mem::transmute(func_ptr) };
        let external_func = ExternalArrayFunc::new(func_ptr);
        PyExternalCall {
            func: ExternalCall::new(external_func, keep_alive),
        }
    }
}

// Python wrapper for Command
#[pyclass(name = "Command")]
#[derive(Clone)]
pub struct PyCommand {
    inner: PyCommandInner,
}

#[derive(Clone)]
enum PyCommandInner {
    Alloc(PyAlloc),
    ExternalCall(PyExternalCall),
}

#[pymethods]
impl PyCommand {
    #[staticmethod]
    fn alloc(shape: Vec<usize>, dtype: PyDType, order: Option<String>) -> PyResult<Self> {
        let alloc = PyAlloc::new(shape, dtype, order)?;
        Ok(PyCommand {
            inner: PyCommandInner::Alloc(alloc),
        })
    }

    #[staticmethod]
    fn external_call(func_ptr: usize, keep_alive: Option<Py<PyAny>>) -> Self {
        let call_func_ptr = PyExternalCall::new(func_ptr, keep_alive);
        PyCommand {
            inner: PyCommandInner::ExternalCall(call_func_ptr),
        }
    }
}

impl PyCommand {
    pub fn to_command(&self) -> Command {
        match &self.inner {
            PyCommandInner::Alloc(alloc) => Command::Alloc(alloc.to_alloc()),
            PyCommandInner::ExternalCall(external_call) => {
                Command::ExternalCall(external_call.func.clone())
            }
        }
    }
}

// Python wrapper for Variable
#[pyclass(name = "Variable", unsendable)]
pub struct PyVariable {
    inner: Variable,
}

#[pymethods]
impl PyVariable {
    #[new]
    fn new_empty_array(rank: usize, dtype: PyDType) -> Self {
        let array = Array::new_empty(rank, dtype.inner);
        let variable = Rc::new(RefCell::new(VariableType::Array(array)));
        PyVariable { inner: variable }
    }

    #[classmethod]
    fn new_array<'py>(
        _cls: &Bound<'py, PyType>,
        py: Python<'py>,
        array: &Bound<'py, numpy::PyUntypedArray>,
        order: Option<String>,
    ) -> Result<Self> {
        let order = order
            .map(|v| Order::from_string(&v))
            .unwrap_or(Ok(Order::c()))
            .context("Invalid order string")?;
        let dtype = array.dtype();

        macro_rules! impl_for_type {
            ($dtype:ty) => {
                if dtype.is_equiv_to(&numpy::dtype::<$dtype>(py)) {
                    let array = array
                        .cast::<PyArrayDyn<$dtype>>()
                        .expect("Failed to cast array")
                        .readonly();
                    let array = Array::from_numpy(array, order)
                        .context("Could not convert array to Variable")?;
                    let variable = Rc::new(RefCell::new(VariableType::Array(array)));
                    return Ok(PyVariable { inner: variable });
                }
            };
        }

        impl_for_type!(f64);
        impl_for_type!(f32);
        impl_for_type!(u64);
        impl_for_type!(i64);
        impl_for_type!(u32);
        impl_for_type!(i32);
        impl_for_type!(u16);
        impl_for_type!(i16);
        impl_for_type!(u8);
        impl_for_type!(i8);
        // TODO figure out what to do with bool arrays

        Err(anyhow::anyhow!("Unsupported dtype: {}", dtype.str()?))
    }

    fn is_array(&self) -> bool {
        matches!(*self.inner.borrow(), VariableType::Array(_))
    }

    fn is_empty_array(&self) -> bool {
        matches!(
            *self.inner.borrow(),
            VariableType::Array(Array {
                buffer: Buffer::Empty,
                ..
            })
        )
    }

    fn is_numba_array(&self) -> bool {
        matches!(
            *self.inner.borrow(),
            VariableType::Array(Array {
                buffer: Buffer::Numba(_),
                ..
            })
        )
    }

    fn is_native_array(&self) -> bool {
        matches!(
            *self.inner.borrow(),
            VariableType::Array(Array {
                buffer: Buffer::Native(_),
                ..
            })
        )
    }

    fn shape(&self) -> Option<Vec<usize>> {
        match &*self.inner.borrow() {
            VariableType::Array(array) => Some(array.shape.to_vec()),
        }
    }

    fn dtype(&self) -> Option<PyDType> {
        let dtype = match &*self.inner.borrow() {
            VariableType::Array(array) => Some(array.dtype),
        };
        let Some(dtype) = dtype else {
            return None;
        };
        Some(PyDType { inner: dtype })
    }

    fn values(&self) -> Result<Option<Py<PyUntypedArray>>> {
        match &*self.inner.borrow() {
            VariableType::Array(array) => array.to_numpy(),
        }
    }
}

// Python wrapper for CommandWithVariables
#[pyclass(name = "CommandWithVariables")]
pub struct PyCommandWithVariables {
    command: PyCommand,
    variables: Vec<Py<PyVariable>>,
}

impl Clone for PyCommandWithVariables {
    fn clone(&self) -> Self {
        Python::attach(|py| {
            let variables = self.variables.iter().map(|var| var.clone_ref(py)).collect();
            PyCommandWithVariables {
                command: self.command.clone(),
                variables,
            }
        })
    }
}

#[pymethods]
impl PyCommandWithVariables {
    #[new]
    fn new(command: PyCommand, variables: Vec<Py<PyVariable>>) -> Self {
        PyCommandWithVariables { command, variables }
    }

    #[staticmethod]
    fn alloc(
        shape: Vec<usize>,
        dtype: PyDType,
        variable: Py<PyVariable>,
        order: Option<String>,
    ) -> PyResult<Self> {
        let command = PyCommand::alloc(shape, dtype, order)?;
        Ok(PyCommandWithVariables::new(command, vec![variable]))
    }

    #[staticmethod]
    fn external_call(
        func_ptr: usize,
        variables: Vec<Py<PyVariable>>,
        keep_alive: Option<Py<PyAny>>,
    ) -> Self {
        let command = PyCommand::external_call(func_ptr, keep_alive);
        PyCommandWithVariables::new(command, variables)
    }

    #[getter]
    fn variables(&self) -> PyResult<Vec<Py<PyVariable>>> {
        Python::attach(|py| {
            let mut result = Vec::new();
            for var in &self.variables {
                result.push(var.clone_ref(py));
            }
            Ok(result)
        })
    }
}

impl PyCommandWithVariables {
    pub fn to_command_with_variables(&self, py: Python) -> Result<CommandWithVariables> {
        let variables: Vec<Variable> = self
            .variables
            .iter()
            .map(|py_var| py_var.bind(py).borrow().inner.clone())
            .collect();

        Ok(CommandWithVariables {
            command: self.command.to_command(),
            variables,
        })
    }
}

// Python wrapper for VM
#[pyclass(name = "VM", unsendable)]
pub struct PyVM {
    inner: VM,
}

#[pymethods]
impl PyVM {
    #[new]
    fn new<'py>(
        py: Python<'py>,
        nrt_api_ptr: usize,
        commands: Vec<PyCommandWithVariables>,
    ) -> Result<Self> {
        // Cast the pointer to the NRT API struct
        let nrt_api = nrt_api_ptr as *const NumbaRuntime;

        // Dereference to get the actual NumbaRuntime struct
        // SAFETY: We trust the user to provide a valid pointer
        let numba_runtime = unsafe { *nrt_api };

        let rust_commands: Result<Vec<_>, _> = commands
            .iter()
            .map(|cmd| cmd.to_command_with_variables(py))
            .collect();

        let rust_commands = rust_commands
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let vm = VM::new(Arc::new(numba_runtime), rust_commands);
        Ok(PyVM { inner: vm })
    }

    fn run(&mut self) -> Result<()> {
        Ok(self.inner.run().context("Failed to run VM")?)
    }
}

// Python builder for convenience
#[pyclass(name = "VMBuilder")]
pub struct PyVMBuilder {
    commands: Vec<PyCommandWithVariables>,
}

#[pymethods]
impl PyVMBuilder {
    #[new]
    fn new() -> Self {
        PyVMBuilder {
            commands: Vec::new(),
        }
    }

    fn add_alloc(
        &mut self,
        shape: Vec<usize>,
        dtype: PyDType,
        variable: Py<PyVariable>,
        order: Option<String>,
    ) -> PyResult<()> {
        let cmd = PyCommandWithVariables::alloc(shape, dtype, variable, order)?;
        self.commands.push(cmd);
        Ok(())
    }

    fn add_external_call(
        &mut self,
        func_ptr: usize,
        variables: Vec<Py<PyVariable>>,
        keep_alive: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let cmd = PyCommandWithVariables::external_call(func_ptr, variables, keep_alive);
        self.commands.push(cmd);
        Ok(())
    }

    fn build<'py>(&self, py: Python<'py>, nrt_api_ptr: usize) -> Result<PyVM> {
        let commands = self.commands.iter().cloned().collect();
        let vm = PyVM::new(py, nrt_api_ptr, commands)?;
        Ok(vm)
    }

    fn num_commands(&self) -> usize {
        self.commands.len()
    }

    fn get_commands(&self) -> PyResult<Vec<Py<PyCommandWithVariables>>> {
        Python::attach(|py| {
            let py_commands: Vec<Py<PyCommandWithVariables>> = self
                .commands
                .iter()
                .map(|cmd| {
                    let cloned_cmd = PyCommandWithVariables {
                        command: cmd.command.clone(),
                        variables: cmd.variables.iter().map(|v| v.clone_ref(py)).collect(),
                    };
                    Py::new(py, cloned_cmd).unwrap()
                })
                .collect();
            Ok(py_commands)
        })
    }
}

// Module definition
#[pymodule]
fn pytensor_rt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDType>()?;
    m.add_class::<PyAlloc>()?;
    m.add_class::<PyExternalCall>()?;
    m.add_class::<PyCommand>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<PyCommandWithVariables>()?;
    m.add_class::<PyVM>()?;
    m.add_class::<PyVMBuilder>()?;
    Ok(())
}
