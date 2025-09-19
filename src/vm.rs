use std::{cell::RefCell, rc::Rc, sync::Arc};

use anyhow::Result;
use pyo3::{Py, PyAny, Python};

use crate::{
    array::{Array, Buffer, DType, NativeBuffer, Order, Shape},
    ffi::{CArgBuffer, ExternalArrayFunc, NumbaRuntime},
};

pub enum VariableType {
    Array(Array),
}

pub type Variable = Rc<RefCell<VariableType>>;

#[derive(Clone, Debug)]
pub struct Alloc {
    shape: Shape,
    dtype: DType,
    order: Order,
}

impl Alloc {
    pub fn new(shape: Shape, dtype: DType, order: Order) -> Self {
        Self {
            shape,
            dtype,
            order,
        }
    }
}

impl CommandTrait for Alloc {
    fn execute(
        &self,
        _vm: &VM,
        _c_arg_buffer: &mut CArgBuffer,
        variables: &mut [Variable],
    ) -> Result<()> {
        for variable in variables.iter() {
            let mut var_borrow = variable.borrow_mut();
            let dtype = self.dtype;

            let item_count = self.shape.iter().fold(1usize, |acc, &dim| {
                acc.checked_mul(dim)
                    .expect("Shape dimensions overflow when calculating item count")
            });
            let _nbytes = item_count
                .checked_mul(dtype.size_bytes())
                .expect("nbytes overflow");

            match &mut *var_borrow {
                VariableType::Array(array) => {
                    match &mut array.buffer {
                        Buffer::Empty => {}
                        Buffer::Native(_buffer) => {
                            if array.dtype == dtype {
                                // TODO: We can reuse the buffer if it has enough bytes
                                //buffer.resize(&self.shape, self.order);
                                //continue;
                            }
                        }
                        Buffer::Numba(_buffer) => {
                            // TODO: We can reuse the buffer if it has enough bytes
                            //if buffer.dtype == dtype && buffer.nbytes == nbytes {
                            //    buffer.reset(&self.shape, self.order);
                            //    continue;
                            //}
                        }
                    };

                    // TODO: wrap in a function in Array
                    let buffer = NativeBuffer::new(dtype, self.shape.as_slice(), self.order);
                    array.buffer = Buffer::Native(buffer);
                    array.shape.clear();
                    array.shape.extend_from_slice(&self.shape);
                }
            }
        }

        Ok(())
    }
}

pub struct ExternalCall {
    external_fn: ExternalArrayFunc,
    keep_alive: Option<Py<PyAny>>,
}

impl Clone for ExternalCall {
    fn clone(&self) -> Self {
        let keep_alive = self
            .keep_alive
            .as_ref()
            .map(|py_any| Python::attach(|py| py_any.clone_ref(py)));

        Self {
            external_fn: self.external_fn.clone(),
            keep_alive,
        }
    }
}

impl ExternalCall {
    pub fn new(external_fn: ExternalArrayFunc, keep_alive: Option<Py<PyAny>>) -> Self {
        Self {
            external_fn,
            keep_alive,
        }
    }
}

impl CommandTrait for ExternalCall {
    fn execute(
        &self,
        vm: &VM,
        c_arg_buffer: &mut CArgBuffer,
        variables: &mut [Variable],
    ) -> Result<()> {
        c_arg_buffer.clear();

        // Add all variables to the command buffer
        for variable in variables.iter() {
            match &*variable.borrow() {
                VariableType::Array(array) => {
                    c_arg_buffer.push_buffer_from_info(array.buffer_info());
                }
            }
        }

        self.external_fn.call(c_arg_buffer).map_err(|e| {
            anyhow::anyhow!("Error calling external function pointer: {}", e.to_string())
        })?;

        // Iterate over modified buffers and update the corresponding variables
        // All modified buffers are guaranteed to be NumbaBuffers.
        variables
            .iter()
            .zip(c_arg_buffer.buffers())
            .for_each(|(variable, buffer_info)| {
                if buffer_info.is_modified() {
                    let mut var_borrow = variable.borrow_mut();
                    match &mut *var_borrow {
                        VariableType::Array(array) => {
                            array.buffer =
                                Buffer::Numba(buffer_info.to_buffer(vm.numba_runtime.clone()));
                        }
                    }
                }
            });

        Ok(())
    }
}

/// A Command modifies the variables in some way.
/// It can allocate new buffers, reshape existing ones,
/// or call a function pointer with the buffers as arguments.
///
/// The caller of the VM is responsible for ensuring
/// that the commands are valid, and that the variables
/// are correctly set up, and memory is not reused in an
/// invalid way.
pub trait CommandTrait: Clone {
    fn execute(
        &self,
        vm: &VM,
        c_arg_buffer: &mut CArgBuffer,
        variables: &mut [Variable],
    ) -> Result<()>;
}

#[non_exhaustive]
#[derive(Clone)]
pub enum Command {
    Alloc(Alloc),
    ExternalCall(ExternalCall),
}

impl CommandTrait for Command {
    fn execute(
        &self,
        vm: &VM,
        c_arg_buffer: &mut CArgBuffer,
        variables: &mut [Variable],
    ) -> Result<()> {
        match self {
            Command::Alloc(alloc) => alloc.execute(vm, c_arg_buffer, variables),
            Command::ExternalCall(call_func_ptr) => {
                call_func_ptr.execute(vm, c_arg_buffer, variables)
            }
        }
    }
}

pub struct CommandWithVariables {
    pub command: Command,
    pub variables: Vec<Variable>,
}

pub struct VM {
    numba_runtime: Arc<NumbaRuntime>,
    commands: Vec<CommandWithVariables>,
}

impl VM {
    pub fn new(numba_runtime: Arc<NumbaRuntime>, commands: Vec<CommandWithVariables>) -> Self {
        Self {
            numba_runtime,
            commands,
        }
    }

    pub fn run(&mut self) -> Result<()> {
        let mut c_arg_buffer = CArgBuffer::new();
        let mut command_variables = Vec::new();
        for CommandWithVariables { command, variables } in &self.commands {
            command_variables.clear();
            command_variables.extend(variables.iter().map(|var| var.clone()));
            command.execute(self, &mut c_arg_buffer, &mut command_variables)?;
        }
        Ok(())
    }
}
