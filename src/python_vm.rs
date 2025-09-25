use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use crate::{
    array::{DType, Order, TensorType},
    ffi::ExternalArrayFunc,
    vm::{
        FunctionId, Module, ModuleBuilder, NodeId, RegionBuilder, RegionId, ValueId, ValueType,
        Variable,
    },
    FunctionBuilder,
};

#[pyclass(name = "TensorType")]
#[derive(Clone)]
pub struct PyTensorType {
    inner: TensorType,
}

#[pymethods]
impl PyTensorType {
    #[new]
    pub fn new(dtype: DType, rank: usize, order: Order) -> PyResult<Self> {
        let tensor_type = TensorType { dtype, rank, order };
        Ok(Self { inner: tensor_type })
    }

    #[getter]
    pub fn rank(&self) -> usize {
        self.inner.rank
    }

    #[getter]
    pub fn dtype(&self) -> String {
        self.inner.dtype.to_string()
    }

    #[getter]
    pub fn order(&self) -> String {
        self.inner.order.to_string()
    }
}

impl<'py> FromPyObject<'py> for DType {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let ob_str = ob.str()?;
        let dtype_str: &str = ob_str.to_str()?;
        Ok(DType::from_string(dtype_str).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid DType string: {}", dtype_str))
        })?)
    }
}

impl<'py> FromPyObject<'py> for Order {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let ob_str = ob.str()?;
        let order_str: &str = ob_str.to_str()?;
        Ok(Order::from_string(order_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid Order string: {}", e))
        })?)
    }
}

#[pyclass(name = "ModuleBuilder")]
pub struct PyModuleBuilder {
    inner: Arc<Mutex<ModuleBuilder>>,
}

#[pymethods]
impl PyModuleBuilder {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ModuleBuilder::new())),
        }
    }

    pub fn add_global_tensor(
        &self,
        name: Option<String>,
        tensor_type: PyTensorType,
    ) -> PyResult<PyValueId> {
        let mut builder = self.inner.lock().unwrap();
        let value_id = builder.add_global_tensor(name, tensor_type.inner);
        Ok(PyValueId(value_id))
    }

    // TODO Should take input/output types, not variables
    // PyVariable shouldn't exist?
    pub fn create_function(
        &self,
        name: String,
        inputs: Vec<PyVariable>,
        outputs: Vec<PyVariable>,
    ) -> PyFunctionBuilder {
        let input_vars: Vec<Variable> = inputs.into_iter().map(|v| v.inner).collect();
        let output_vars: Vec<Variable> = outputs.into_iter().map(|v| v.inner).collect();

        let mut builder = self.inner.lock().unwrap();
        let function_builder = builder.create_function(name, input_vars, output_vars);

        PyFunctionBuilder {
            inner: Some(function_builder),
            module_builder: self.inner.clone(),
        }
    }

    pub fn build(&self) -> PyTensorModule {
        let builder = std::mem::replace(&mut *self.inner.lock().unwrap(), ModuleBuilder::new());
        let module = builder.build();
        PyTensorModule(module)
    }
}

#[pyclass(name = "FunctionBuilder")]
pub struct PyFunctionBuilder {
    inner: Option<FunctionBuilder>,
    module_builder: Arc<Mutex<ModuleBuilder>>,
}

#[pymethods]
impl PyFunctionBuilder {
    pub fn create_region(&mut self) -> PyResult<PyRegionBuilder> {
        let function_builder = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Function already finished")
        })?;

        let region_builder = function_builder.create_region(self.module_builder.clone());
        Ok(PyRegionBuilder(Some(region_builder)))
    }

    pub fn finish(&mut self, entry_region: PyRegionId) -> PyResult<PyFunctionId> {
        let function_builder = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Function already finished")
        })?;

        let function_id = function_builder.finish(self.module_builder.clone(), entry_region.0);
        Ok(PyFunctionId(function_id))
    }

    pub fn input(&self, idx: usize) -> PyResult<PyValueId> {
        let function_builder = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Function already finished")
        })?;

        let Some(input) = function_builder.input(idx) else {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Input index out of range",
            ));
        };
        Ok(PyValueId(input))
    }

    pub fn inputs(&self) -> PyResult<Vec<PyValueId>> {
        let function_builder = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Function already finished")
        })?;

        (0..function_builder.num_inputs())
            .map(|i| {
                function_builder
                    .input(i)
                    .map(|id| PyValueId(id))
                    .ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err("Input index out of range")
                    })
            })
            .collect()
    }

    pub fn output(&self, idx: usize) -> PyResult<PyValueId> {
        let function_builder = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Function already finished")
        })?;

        let Some(output) = function_builder.output(idx) else {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Output index out of range",
            ));
        };
        Ok(PyValueId(output))
    }

    pub fn outputs(&self) -> PyResult<Vec<PyValueId>> {
        let function_builder = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Function already finished")
        })?;

        (0..function_builder.num_outputs())
            .map(|i| {
                function_builder
                    .output(i)
                    .map(|id| PyValueId(id))
                    .ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err("Output index out of range")
                    })
            })
            .collect()
    }
}

#[pyclass(name = "RegionBuilder")]
pub struct PyRegionBuilder(Option<RegionBuilder>);

#[pymethods]
impl PyRegionBuilder {
    pub fn add_alloc(
        &mut self,
        target: PyValueId,
        shape: PyValueId,
        tensor_type: PyTensorType,
    ) -> PyResult<PyNodeId> {
        let region_builder = self
            .0
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Region already finished"))?;

        let node_id = region_builder.add_alloc(target.0, shape.0, tensor_type.inner);
        Ok(PyNodeId(node_id))
    }

    pub fn add_external_call(
        &mut self,
        func_ptr: isize,
        args: Vec<PyValueId>,
        arg_is_mut: Vec<bool>,
        name: String,
    ) -> PyResult<PyNodeId> {
        let region_builder = self
            .0
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Region already finished"))?;

        // Create ExternalArrayFunc from raw pointer
        // SAFETY: This is unsafe - the caller must ensure the function pointer is valid
        let func = ExternalArrayFunc::new(unsafe { std::mem::transmute(func_ptr) });

        let args: Vec<ValueId> = args.into_iter().map(|v| v.0).collect();
        let node_id = region_builder.add_external_call(func, None, args, arg_is_mut, name);
        Ok(PyNodeId(node_id))
    }

    pub fn add_if(
        &mut self,
        condition: PyValueId,
        then_region: PyRegionId,
        else_region: PyRegionId,
    ) -> PyResult<PyNodeId> {
        let region_builder = self
            .0
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Region already finished"))?;

        let node_id = region_builder.add_if(condition.0, then_region.0, else_region.0);
        Ok(PyNodeId(node_id))
    }

    pub fn add_while(
        &mut self,
        condition: PyValueId,
        cond_region: PyRegionId,
        body_region: PyRegionId,
    ) -> PyResult<PyNodeId> {
        let region_builder = self
            .0
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Region already finished"))?;

        let node_id = region_builder.add_while(condition.0, cond_region.0, body_region.0);
        Ok(PyNodeId(node_id))
    }

    pub fn add_dependency(&mut self, from: PyNodeId, to: PyNodeId) -> PyResult<()> {
        let region_builder = self
            .0
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Region already finished"))?;

        region_builder.add_dependency(from.0, to.0);
        Ok(())
    }

    pub fn create_subregion(&self) -> PyResult<PyRegionBuilder> {
        let region_builder = self
            .0
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Region already finished"))?;

        let subregion = region_builder.create_subregion();
        Ok(PyRegionBuilder(Some(subregion)))
    }

    pub fn finish(&mut self) -> PyResult<(Option<PyFunctionBuilder>, PyRegionId)> {
        let region_builder = self
            .0
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Region already finished"))?;

        let (function_builder, region_id) = region_builder.finish();
        let py_function_builder = function_builder.map(|fb| PyFunctionBuilder {
            inner: Some(fb),
            module_builder: Arc::new(Mutex::new(ModuleBuilder::new())), // This is a bit of a hack
        });

        Ok((py_function_builder, PyRegionId(region_id)))
    }
}

#[pyclass(name = "ValueId")]
#[derive(Clone)]
pub struct PyValueId(ValueId);

#[pyclass(name = "NodeId")]
#[derive(Clone)]
pub struct PyNodeId(NodeId);

#[pyclass(name = "RegionId")]
#[derive(Clone)]
pub struct PyRegionId(RegionId);

#[pyclass(name = "RegionId")]
#[derive(Clone)]
pub struct PyFunctionId(FunctionId);

#[pyclass(name = "Variable")]
#[derive(Clone)]
pub struct PyVariable {
    inner: Variable,
}

#[pymethods]
impl PyVariable {
    #[new]
    pub fn new(tensor_type: PyTensorType, name: Option<String>) -> PyResult<Self> {
        let variable = Variable::new(ValueType::Tensor(tensor_type.inner), name);
        Ok(Self { inner: variable })
    }
}

#[pyclass(name = "Module")]
pub struct PyTensorModule(Module);

#[pymethods]
impl PyTensorModule {
    pub fn __str__(&self) -> String {
        format!("{}", self.0)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PyTensorModule(functions={}, globals={})",
            self.0.num_functions(),
            self.0.num_variables()
        )
    }

    pub fn num_functions(&self) -> usize {
        self.0.num_functions()
    }

    pub fn num_variables(&self) -> usize {
        self.0.num_variables()
    }

    pub fn num_regions(&self) -> usize {
        self.0.num_regions()
    }

    pub fn num_nodes(&self) -> usize {
        self.0.num_nodes()
    }

    pub fn asm(&self) -> String {
        self.0.to_string()
    }
}

pub fn create_python_module(_py: Python<'_>, m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PyModuleBuilder>()?;
    m.add_class::<PyFunctionBuilder>()?;
    m.add_class::<PyRegionBuilder>()?;
    m.add_class::<PyValueId>()?;
    m.add_class::<PyNodeId>()?;
    m.add_class::<PyRegionId>()?;
    m.add_class::<PyFunctionId>()?;
    m.add_class::<PyTensorType>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<PyTensorModule>()?;
    Ok(())
}
