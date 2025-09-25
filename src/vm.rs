use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
    sync::{atomic::AtomicUsize, Arc, Mutex, RwLock},
};

use crossbeam_utils::CachePadded;
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use smallvec::SmallVec;
use thiserror::Error;

use crate::{
    array::TensorType, ffi::ExternalCallBuffer, Buffer, ExternalArrayFunc, NumbaRuntime, Tensor,
};

#[derive(Error, Debug)]
pub enum Error {
    #[error("External function call failed: {message}")]
    ExternalError { message: String },

    #[error("Invalid graph: {kind}")]
    InvalidGraphError { kind: InvalidGraphKind },

    #[error("Value error: {message}")]
    ValueError { message: String },

    #[error("Tensor operation failed: {0}")]
    TensorError(#[from] anyhow::Error),
}

#[derive(Error, Debug)]
pub enum InvalidGraphKind {
    #[error(
        "Wrong number of arguments: instruction '{instruction}' expected {expected}, got {actual}"
    )]
    WrongArgumentCount {
        instruction: String,
        expected: usize,
        actual: usize,
    },

    #[error("Empty value in {context}")]
    EmptyValue { context: String },

    #[error("Invalid value type in {context}")]
    InvalidValueType { context: String },
}

pub type Result<T> = std::result::Result<T, Error>;

thread_local! {
    static EXTERNAL_CALL_BUFFER: RefCell<ExternalCallBuffer> = const { RefCell::new(ExternalCallBuffer::new()) };
}

#[derive(Debug)]
pub enum Value {
    Tensor(Tensor),
    Empty,
}

impl Value {
    fn as_scalar_bool(&self) -> Result<bool> {
        match self {
            Value::Tensor(t) => Ok(t.as_scalar_bool()?),
            Value::Empty => Err(Error::ValueError {
                message: "Value is empty".to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalValueId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueId {
    Input(FunctionId, usize),
    Output(FunctionId, usize),
    Global(GlobalValueId),
}

#[derive(Debug, Clone)]
pub enum ValueType {
    Tensor(TensorType),
    // Could add other types later like Counter, Bool, etc.
}

#[derive(Debug, Clone)]
pub struct Variable {
    pub value_type: ValueType,
    pub name: Option<String>,
}

impl Variable {
    pub fn new(value_type: ValueType, name: Option<String>) -> Self {
        Self { value_type, name }
    }
}

#[derive(Debug)]
struct Node {
    args: Vec<ValueId>,
    successors: Vec<NodeId>,
    predecessors: Vec<NodeId>,
    instruction: Instruction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl Node {
    fn run<C: ExecutionContext>(&self, ctx: &C, store: &C::ValueStore) -> Result<()> {
        self.instruction.run(ctx, store, &self.args)
    }
}

#[derive(Debug)]
pub struct Region {
    nodes: Vec<NodeId>,
    sequential: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(usize);

pub struct ExternalCall {
    func: ExternalArrayFunc,
    _keep_alive: Option<Box<dyn std::any::Any + Send + Sync>>,
    args: Vec<ValueId>,
    arg_is_mut: Vec<bool>,
    name: String,
}

impl Debug for ExternalCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalCall")
            .field("func", &self.func)
            .field("args", &self.args)
            .field("arg_is_mut", &self.arg_is_mut)
            .field("name", &self.name)
            .finish()
    }
}

// SAFETY: ExternalCall can be sent between threads as long as the function pointer and
// keep_alive are Send + Sync.
// We require the function pointer passed by upstream libraries to be Send.
unsafe impl Send for ExternalCall {}

impl RunInstruction for ExternalCall {
    fn run<C: ExecutionContext>(
        &self,
        ctx: &C,
        store: &C::ValueStore,
        args: &[ValueId],
    ) -> Result<()> {
        if args.len() != self.args.len() {
            return Err(Error::InvalidGraphError {
                kind: InvalidGraphKind::WrongArgumentCount {
                    instruction: format!("ExternalCall({})", self.name),
                    expected: self.args.len(),
                    actual: args.len(),
                },
            });
        }
        EXTERNAL_CALL_BUFFER.with_borrow_mut(|c_arg_buffer| {
            c_arg_buffer.clear();

            let mut vals_mut: SmallVec<[_; 8]> = SmallVec::new();
            let mut vals: SmallVec<[_; 8]> = SmallVec::new();
            for (&arg_id, &is_mut) in args.iter().zip_eq(self.arg_is_mut.iter()) {
                if is_mut {
                    let val = store.resolve_value_mut(arg_id);
                    match &*val {
                        Value::Tensor(tensor) => {
                            let info = tensor.buffer_info();
                            c_arg_buffer.push_buffer_from_info(info);
                        }
                        Value::Empty => {
                            return Err(Error::InvalidGraphError {
                                kind: InvalidGraphKind::EmptyValue {
                                    context: "mutable argument in external call".to_string(),
                                },
                            });
                        }
                    }
                    vals_mut.push(val);
                } else {
                    let val = store.resolve_value(arg_id);
                    match &*val {
                        Value::Tensor(tensor) => {
                            let info = tensor.buffer_info();
                            c_arg_buffer.push_buffer_from_info(info);
                        }
                        Value::Empty => {
                            return Err(Error::InvalidGraphError {
                                kind: InvalidGraphKind::EmptyValue {
                                    context: "immutable argument in external call".to_string(),
                                },
                            });
                        }
                    }
                    vals.push(val);
                }
            }
            let _ = self
                .func
                .call(c_arg_buffer)
                .map_err(|e| Error::ExternalError {
                    message: format!("Error calling external function pointer: {}", e),
                })?;

            let mut modified_idx = 0;
            for (buffer_info, &is_mut) in c_arg_buffer.buffers().zip_eq(self.arg_is_mut.iter()) {
                if is_mut {
                    let val = &mut vals_mut[modified_idx];
                    modified_idx += 1;
                    match val.deref_mut() {
                        Value::Tensor(tensor) => {
                            tensor.buffer =
                                Buffer::Numba(buffer_info.to_buffer(ctx.numba_runtime().clone()))
                        }
                        Value::Empty => {
                            return Err(Error::InvalidGraphError {
                                kind: InvalidGraphKind::EmptyValue {
                                    context: "mutable argument was empty in external call"
                                        .to_string(),
                                },
                            });
                        }
                    }
                } else {
                    if buffer_info.is_modified() {
                        panic!("Immutable argument was modified by external function");
                    }
                }
            }

            // This might unlock the values
            drop(vals_mut);
            drop(vals);
            Ok(())
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// Takes two arguments: target and shape
#[derive(Debug)]
pub struct Alloc {
    tensor_type: TensorType,
}

impl RunInstruction for Alloc {
    fn run<C: ExecutionContext>(
        &self,
        _ctx: &C,
        store: &C::ValueStore,
        args: &[ValueId],
    ) -> Result<()> {
        if args.len() != 2 {
            return Err(Error::InvalidGraphError {
                kind: InvalidGraphKind::WrongArgumentCount {
                    instruction: "Alloc".to_string(),
                    expected: 2,
                    actual: args.len(),
                },
            });
        }
        let shape_value = store.resolve_value(args[1]);
        let shape = match &*shape_value {
            Value::Tensor(t) => t.as_shape()?,
            Value::Empty => {
                return Err(Error::InvalidGraphError {
                    kind: InvalidGraphKind::EmptyValue {
                        context: "shape argument in Alloc".to_string(),
                    },
                });
            }
        };
        let tensor = Tensor::new(
            self.tensor_type.dtype,
            shape.as_slice(),
            self.tensor_type.order,
        );
        let mut target_value = store.resolve_value_mut(args[0]);
        *target_value = Value::Tensor(tensor);
        Ok(())
    }

    fn name(&self) -> &str {
        "Alloc"
    }
}

#[derive(Debug)]
pub struct If {
    then_branch: RegionId,
    else_branch: RegionId,
}

impl RunInstruction for If {
    fn run<C: ExecutionContext>(
        &self,
        ctx: &C,
        store: &C::ValueStore,
        args: &[ValueId],
    ) -> Result<()> {
        if args.len() != 1 {
            return Err(Error::InvalidGraphError {
                kind: InvalidGraphKind::WrongArgumentCount {
                    instruction: "If".to_string(),
                    expected: 1,
                    actual: args.len(),
                },
            });
        }
        let cond_value = store.resolve_value(args[0]);
        let cond = cond_value.as_scalar_bool()?;
        if cond {
            ctx.run_region(self.then_branch, store)
        } else {
            ctx.run_region(self.else_branch, store)
        }
    }

    fn name(&self) -> &str {
        "If"
    }
}

#[derive(Debug)]
pub struct While {
    cond: RegionId,
    body: RegionId,
}

#[derive(Debug)]
pub struct Block {
    region: RegionId,
}

impl RunInstruction for While {
    fn run<C: ExecutionContext>(
        &self,
        ctx: &C,
        store: &C::ValueStore,
        args: &[ValueId],
    ) -> Result<()> {
        if args.len() != 1 {
            return Err(Error::InvalidGraphError {
                kind: InvalidGraphKind::WrongArgumentCount {
                    instruction: "While".to_string(),
                    expected: 1,
                    actual: args.len(),
                },
            });
        }
        loop {
            ctx.run_region(self.cond, store)?;
            let cond_value = store.resolve_value(args[0]);
            let cond = cond_value.as_scalar_bool()?;
            if !cond {
                break;
            }
            ctx.run_region(self.body, store)?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "While"
    }
}

impl RunInstruction for Block {
    fn run<C: ExecutionContext>(
        &self,
        ctx: &C,
        store: &C::ValueStore,
        args: &[ValueId],
    ) -> Result<()> {
        if !args.is_empty() {
            return Err(Error::InvalidGraphError {
                kind: InvalidGraphKind::WrongArgumentCount {
                    instruction: "Block".to_string(),
                    expected: 0,
                    actual: args.len(),
                },
            });
        }
        ctx.run_region(self.region, store)
    }

    fn name(&self) -> &str {
        "Block"
    }
}

pub trait DynInstruction: Send + Sync + Display + Debug {
    fn run_dyn(&self, inputs: &[&Value], outputs: &mut [&mut Value]) -> Result<()>;
    fn num_inputs(&self) -> usize;
    fn num_outputs(&self) -> usize;
    fn name(&self) -> &str;
}

#[non_exhaustive]
#[derive(Debug)]
pub enum Instruction {
    If(If),
    While(While),
    Block(Block),
    ExternalCall(ExternalCall),
    Alloc(Alloc),
    Dyn(Box<dyn DynInstruction>),
}

impl Instruction {
    fn run<C: ExecutionContext>(
        &self,
        ctx: &C,
        store: &C::ValueStore,
        args: &[ValueId],
    ) -> Result<()> {
        match self {
            Instruction::If(inst) => inst.run(ctx, store, args),
            Instruction::While(inst) => inst.run(ctx, store, args),
            Instruction::Block(inst) => inst.run(ctx, store, args),
            Instruction::ExternalCall(inst) => inst.run(ctx, store, args),
            Instruction::Alloc(inst) => inst.run(ctx, store, args),
            Instruction::Dyn(inst) => {
                let num_inputs = inst.num_inputs();
                let num_outputs = inst.num_outputs();
                if args.len() != num_inputs + num_outputs {
                    return Err(Error::InvalidGraphError {
                        kind: InvalidGraphKind::WrongArgumentCount {
                            instruction: inst.name().to_string(),
                            expected: num_inputs + num_outputs,
                            actual: args.len(),
                        },
                    });
                }

                let input_values: SmallVec<[_; 8]> = args[..num_inputs]
                    .iter()
                    .map(|&arg_id| store.resolve_value(arg_id))
                    .collect();

                let mut output_values: SmallVec<[_; 8]> = args[num_inputs..]
                    .iter()
                    .map(|&arg_id| store.resolve_value_mut(arg_id))
                    .collect();

                let input_refs: SmallVec<[_; 8]> = input_values.iter().map(|v| v.deref()).collect();
                let mut output_refs: SmallVec<[_; 8]> =
                    output_values.iter_mut().map(|v| v.deref_mut()).collect();
                inst.run_dyn(&input_refs, &mut output_refs)?;

                Ok(())
            }
        }
    }

    pub fn if_(then_branch: RegionId, else_branch: RegionId) -> Self {
        Instruction::If(If {
            then_branch,
            else_branch,
        })
    }

    pub fn while_(cond: RegionId, body: RegionId) -> Self {
        Instruction::While(While { cond, body })
    }

    pub fn block(region: RegionId) -> Self {
        Instruction::Block(Block { region })
    }

    pub fn external_call(
        func: ExternalArrayFunc,
        keep_alive: Option<Box<dyn std::any::Any + Send + Sync>>,
        args: Vec<ValueId>,
        arg_is_mut: Vec<bool>,
        name: String,
    ) -> Self {
        Instruction::ExternalCall(ExternalCall {
            func,
            _keep_alive: keep_alive,
            args,
            arg_is_mut,
            name,
        })
    }

    pub fn alloc(tensor_type: TensorType) -> Self {
        Instruction::Alloc(Alloc { tensor_type })
    }

    pub fn dyn_instruction(inst: Box<dyn DynInstruction>) -> Self {
        Instruction::Dyn(inst)
    }
}

trait RunInstruction: Debug {
    fn run<C: ExecutionContext>(
        &self,
        ctx: &C,
        store: &C::ValueStore,
        args: &[ValueId],
    ) -> Result<()>;

    fn name(&self) -> &str;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(usize);

#[derive(Debug)]
pub struct Function {
    entry: RegionId,
    name: String,
    id: FunctionId,
    input_variables: Vec<Variable>,
    output_variables: Vec<Variable>,
}

#[derive(Debug)]
pub struct Module {
    regions: Vec<Region>,
    nodes: Vec<Node>,
    functions: Vec<Function>,
    globals: Vec<Variable>,
}

impl Module {
    fn resolve_region(&self, region_id: RegionId) -> &Region {
        &self.regions[region_id.0]
    }

    fn resolve_node(&self, node_id: NodeId) -> &Node {
        &self.nodes[node_id.0]
    }

    fn resolve_function(&self, function_id: FunctionId) -> &Function {
        &self.functions[function_id.0]
    }

    pub fn resolve_global(&self, global_id: GlobalValueId) -> &Variable {
        &self.globals[global_id.0]
    }

    fn resolve_input_variable(&self, func_id: FunctionId, index: usize) -> &Variable {
        &self.functions[func_id.0].input_variables[index]
    }

    fn resolve_output_variable(&self, func_id: FunctionId, index: usize) -> &Variable {
        &self.functions[func_id.0].output_variables[index]
    }

    pub fn resolve_variable(&self, value_id: ValueId) -> &Variable {
        match value_id {
            ValueId::Input(func_id, idx) => self.resolve_input_variable(func_id, idx),
            ValueId::Output(func_id, idx) => self.resolve_output_variable(func_id, idx),
            ValueId::Global(global_id) => self.resolve_global(global_id),
        }
    }

    /// Resolve the type of a value in the context of a specific function
    pub fn resolve_value_type(&self, value_id: ValueId) -> &ValueType {
        match value_id {
            ValueId::Input(func_id, idx) => {
                &self
                    .resolve_function(func_id)
                    .input_variables
                    .get(idx)
                    .expect("Input index out of bounds")
                    .value_type
            }
            ValueId::Output(func_id, idx) => {
                &self
                    .resolve_function(func_id)
                    .output_variables
                    .get(idx)
                    .expect("Output index out of bounds")
                    .value_type
            }
            ValueId::Global(GlobalValueId(idx)) => {
                &self
                    .globals
                    .get(idx)
                    .expect("Global variable index out of bounds")
                    .value_type
            }
        }
    }

    /// Get the number of regions (for Python bindings)
    pub fn num_regions(&self) -> usize {
        self.regions.len()
    }

    /// Get the number of nodes (for Python bindings)
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_functions(&self) -> usize {
        self.functions.len()
    }

    pub fn num_variables(&self) -> usize {
        self.globals.len()
    }
}

impl Display for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "; Module: example.prt")?;
        writeln!(f)?;

        // Write global value declarations
        writeln!(f, "; Global Value declarations with optional names")?;
        for (i, variable) in self.globals.iter().enumerate() {
            if let Some(name) = &variable.name {
                write!(f, "%v_{}: ", name)?;
            } else {
                write!(f, "%v{}: ", i)?;
            }

            match &variable.value_type {
                ValueType::Tensor(tensor_type) => {
                    write!(f, "{}", tensor_type)?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f)?;

        // Write function comments
        writeln!(
            f,
            "; %c* values represent control flow, they order the operations"
        )?;
        writeln!(f, "; %v* values represent storage locations (variables)")?;
        writeln!(f, "; %i* values represent inputs to the function")?;
        writeln!(f, "; %o* values represent outputs of the function")?;

        // Write functions
        for function in &self.functions {
            write!(f, "fn {}(", function.name)?;

            // Write inputs on separate lines if there are any
            if !function.input_variables.is_empty() {
                writeln!(f)?;
                for (idx, input_var) in function.input_variables.iter().enumerate() {
                    write!(f, "    %i{}: ", idx)?;
                    match &input_var.value_type {
                        ValueType::Tensor(tensor_type) => {
                            write!(f, "{}", format_tensor_type(tensor_type))?;
                        }
                    }
                    if idx < function.input_variables.len() - 1 {
                        writeln!(f, ",")?;
                    } else {
                        writeln!(f)?;
                    }
                }
            }

            write!(f, ") -> (")?;

            // Write outputs on separate lines if there are any
            if !function.output_variables.is_empty() {
                writeln!(f)?;
                for (idx, output_var) in function.output_variables.iter().enumerate() {
                    write!(f, "    %o{}: ", idx)?;
                    match &output_var.value_type {
                        ValueType::Tensor(tensor_type) => {
                            write!(f, "{}", format_tensor_type(tensor_type))?;
                        }
                    }
                    if idx < function.output_variables.len() - 1 {
                        writeln!(f, ",")?;
                    } else {
                        writeln!(f)?;
                    }
                }
            }

            write!(f, ")")?;

            let region = self.resolve_region(function.entry);
            if region.sequential {
                write!(f, " [seq]")?;
            }
            writeln!(f, " {{")?;

            self.format_region(f, region, 1)?;

            writeln!(f, "}}")?;
            writeln!(f)?;
        }

        Ok(())
    }
}

impl Module {
    fn format_region(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        region: &Region,
        indent_level: usize,
    ) -> std::fmt::Result {
        let indent = "    ".repeat(indent_level);

        for (i, &node_id) in region.nodes.iter().enumerate() {
            let node = self.resolve_node(node_id);
            write!(f, "{}%c{} = ", indent, i)?;

            match &node.instruction {
                Instruction::Alloc(_) => {
                    write!(f, "Alloc(")?;
                    self.format_predecessors(f, &node.predecessors, region)?;
                    write!(f, "target=")?;
                    self.format_value_id(f, node.args[0])?;
                    write!(f, ", shape=")?;
                    self.format_value_id(f, node.args[1])?;
                    write!(f, ")")?;
                }
                Instruction::ExternalCall(call) => {
                    write!(f, "ExternalCall(")?;
                    self.format_predecessors(f, &node.predecessors, region)?;
                    write!(f, "\"{}\"", call.name)?;
                    for (j, &arg) in node.args.iter().enumerate() {
                        write!(f, ", ")?;
                        if call.arg_is_mut[j] {
                            write!(f, "mut ")?;
                        }
                        self.format_value_id(f, arg)?;
                    }
                    write!(f, ")")?;
                }
                Instruction::If(if_instr) => {
                    write!(f, "If(")?;
                    self.format_predecessors(f, &node.predecessors, region)?;
                    write!(f, "condition=")?;
                    self.format_value_id(f, node.args[0])?;
                    write!(f, ") then")?;

                    let then_region = self.resolve_region(if_instr.then_branch);
                    if then_region.sequential {
                        write!(f, " [seq]")?;
                    }
                    writeln!(f, " {{")?;
                    self.format_region(f, then_region, indent_level + 1)?;

                    write!(f, "{}}} else", indent)?;
                    let else_region = self.resolve_region(if_instr.else_branch);
                    if else_region.sequential {
                        write!(f, " [seq]")?;
                    }
                    writeln!(f, " {{")?;
                    self.format_region(f, else_region, indent_level + 1)?;

                    write!(f, "{}}}", indent)?;
                }
                Instruction::While(while_instr) => {
                    write!(f, "{}(", while_instr.name())?;
                    self.format_predecessors(f, &node.predecessors, region)?;
                    write!(f, "condition=")?;
                    self.format_value_id(f, node.args[0])?;
                    writeln!(f, ") {{")?;

                    let cond_region = self.resolve_region(while_instr.cond);
                    self.format_region(f, cond_region, indent_level + 1)?;

                    let body_region = self.resolve_region(while_instr.body);
                    self.format_region(f, body_region, indent_level + 1)?;

                    write!(f, "{}}}", indent)?;
                }
                Instruction::Block(block_instr) => {
                    write!(f, "Block(")?;
                    self.format_predecessors(f, &node.predecessors, region)?;
                    write!(f, ")")?;

                    let block_region = self.resolve_region(block_instr.region);
                    if block_region.sequential {
                        write!(f, " [seq]")?;
                    }
                    writeln!(f, " {{")?;
                    self.format_region(f, block_region, indent_level + 1)?;

                    write!(f, "{}}}", indent)?;
                }
                Instruction::Dyn(dyn_instruction) => {
                    write!(f, "{}(", dyn_instruction.name())?;
                    self.format_predecessors(f, &node.predecessors, region)?;
                    for (j, &arg) in node.args.iter().enumerate() {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        self.format_value_id(f, arg)?;
                    }
                    write!(f, ")")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }

    fn format_predecessors(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        predecessors: &[NodeId],
        region: &Region,
    ) -> std::fmt::Result {
        if !predecessors.is_empty() {
            let pred_indices: Vec<usize> = predecessors
                .iter()
                .map(|&pred_id| {
                    region
                        .nodes
                        .iter()
                        .position(|&nid| nid == pred_id)
                        .unwrap_or(0)
                })
                .collect();

            for (i, idx) in pred_indices.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "%c{}", idx)?;
            }
            write!(f, "; ")?;
        }
        Ok(())
    }

    fn format_value_id(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        value_id: ValueId,
    ) -> std::fmt::Result {
        match value_id {
            ValueId::Global(GlobalValueId(idx)) => {
                if let Some(variable) = self.globals.get(idx) {
                    if let Some(name) = &variable.name {
                        write!(f, "%g_{}", name)?;
                    } else {
                        write!(f, "%g{}", idx)?;
                    }
                } else {
                    write!(f, "%g{}", idx)?;
                }
            }
            ValueId::Input(_, idx) => write!(f, "%i{}", idx)?,
            ValueId::Output(_, idx) => write!(f, "%o{}", idx)?,
        }
        Ok(())
    }
}

fn format_tensor_type(tensor_type: &TensorType) -> String {
    tensor_type.to_string()
}

// Builder pattern implementation
#[derive(Debug)]
pub struct ModuleBuilder {
    regions: Vec<Region>,
    nodes: Vec<Node>,
    functions: Vec<Function>,
    globals: Vec<Variable>,
    next_region_id: usize,
    next_node_id: usize,
    next_function_id: usize,
    next_global_id: usize,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            nodes: Vec::new(),
            functions: Vec::new(),
            globals: Vec::new(),
            next_region_id: 0,
            next_node_id: 0,
            next_function_id: 0,
            next_global_id: 0,
        }
    }

    pub fn build(self) -> Module {
        Module {
            regions: self.regions,
            nodes: self.nodes,
            functions: self.functions,
            globals: self.globals,
        }
    }

    pub fn add_global(&mut self, variable: Variable) -> GlobalValueId {
        let id = GlobalValueId(self.next_global_id);
        self.next_global_id += 1;
        self.globals.push(variable);
        id
    }

    pub fn add_global_tensor(&mut self, name: Option<String>, tensor_type: TensorType) -> ValueId {
        let variable = Variable::new(ValueType::Tensor(tensor_type), name);
        let global_id = self.add_global(variable);
        ValueId::Global(global_id)
    }

    pub fn create_function(
        &mut self,
        name: String,
        inputs: Vec<Variable>,
        outputs: Vec<Variable>,
    ) -> FunctionBuilder {
        let function_id = FunctionId(self.next_function_id);
        self.next_function_id += 1;

        FunctionBuilder {
            function_id,
            name,
            input_variables: inputs,
            output_variables: outputs,
            entry_graph: None,
        }
    }

    fn add_region(&mut self, region: Region) -> RegionId {
        let id = RegionId(self.next_region_id);
        self.next_region_id += 1;
        self.regions.push(region);
        id
    }

    fn add_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        self.nodes.push(node);
        id
    }

    fn add_function(&mut self, function: Function) -> FunctionId {
        let id = function.id;
        self.functions.push(function);
        id
    }
}

#[derive(Debug, Clone)]
pub struct FunctionBuilder {
    function_id: FunctionId,
    name: String,
    input_variables: Vec<Variable>,
    output_variables: Vec<Variable>,
    entry_graph: Option<RegionId>,
}

impl FunctionBuilder {
    pub fn create_region(&self, module_builder: Arc<Mutex<ModuleBuilder>>) -> RegionBuilder {
        RegionBuilder::new(module_builder, Some(self.clone()))
    }

    pub fn finish(
        mut self,
        module_builder: Arc<Mutex<ModuleBuilder>>,
        entry_graph: RegionId,
    ) -> FunctionId {
        self.entry_graph = Some(entry_graph);

        let function = Function {
            entry: entry_graph,
            name: self.name,
            id: self.function_id,
            input_variables: self.input_variables,
            output_variables: self.output_variables,
        };

        let mut builder = module_builder.lock().unwrap();
        builder.add_function(function)
    }

    pub fn function_id(&self) -> FunctionId {
        self.function_id
    }

    pub fn input(&self, idx: usize) -> Option<ValueId> {
        if idx >= self.input_variables.len() {
            return None;
        }
        Some(ValueId::Input(self.function_id, idx))
    }

    pub fn output(&self, idx: usize) -> Option<ValueId> {
        if idx >= self.output_variables.len() {
            return None;
        }
        Some(ValueId::Output(self.function_id, idx))
    }

    pub fn num_inputs(&self) -> usize {
        self.input_variables.len()
    }

    pub fn num_outputs(&self) -> usize {
        self.output_variables.len()
    }
}

#[derive(Debug)]
pub struct RegionBuilder {
    module_builder: Arc<Mutex<ModuleBuilder>>,
    nodes: Vec<NodeId>,
    function_builder: Option<FunctionBuilder>,
    sequential: bool,
}

impl RegionBuilder {
    pub fn new(
        module_builder: Arc<Mutex<ModuleBuilder>>,
        function_builder: Option<FunctionBuilder>,
    ) -> Self {
        Self {
            module_builder,
            nodes: Vec::new(),
            function_builder,
            sequential: false,
        }
    }

    pub fn add_node(&mut self, instruction: Instruction, args: Vec<ValueId>) -> NodeId {
        let node = Node {
            args,
            successors: Vec::new(),
            predecessors: Vec::new(),
            instruction,
        };

        let mut builder = self.module_builder.lock().unwrap();
        let node_id = builder.add_node(node);
        self.nodes.push(node_id);
        node_id
    }

    pub fn add_dependency(&mut self, from: NodeId, to: NodeId) {
        let mut builder = self.module_builder.lock().unwrap();
        builder.nodes[from.0].successors.push(to);
        builder.nodes[to.0].predecessors.push(from);
    }

    pub fn finish(self) -> (Option<FunctionBuilder>, RegionId) {
        let region = Region {
            nodes: self.nodes,
            sequential: self.sequential,
        };

        let mut builder = self.module_builder.lock().unwrap();
        let region_id = builder.add_region(region);
        (self.function_builder, region_id)
    }

    pub fn create_subregion(&self) -> RegionBuilder {
        RegionBuilder::new(self.module_builder.clone(), None)
    }

    pub fn sequential(mut self) -> Self {
        self.sequential = true;
        self
    }

    pub fn parallel(mut self) -> Self {
        self.sequential = false;
        self
    }

    pub fn add_alloc(
        &mut self,
        target: ValueId,
        shape: ValueId,
        tensor_type: TensorType,
    ) -> NodeId {
        let instruction = Instruction::alloc(tensor_type);
        self.add_node(instruction, vec![target, shape])
    }

    pub fn add_external_call(
        &mut self,
        func: ExternalArrayFunc,
        keep_alive: Option<Box<dyn std::any::Any + Send + Sync>>,
        args: Vec<ValueId>,
        arg_is_mut: Vec<bool>,
        name: String,
    ) -> NodeId {
        let instruction =
            Instruction::external_call(func, keep_alive, args.clone(), arg_is_mut, name);
        self.add_node(instruction, args)
    }

    pub fn add_if(
        &mut self,
        condition: ValueId,
        then_region: RegionId,
        else_region: RegionId,
    ) -> NodeId {
        let instruction = Instruction::if_(then_region, else_region);
        self.add_node(instruction, vec![condition])
    }

    pub fn add_while(
        &mut self,
        condition: ValueId,
        cond_region: RegionId,
        body_region: RegionId,
    ) -> NodeId {
        let instruction = Instruction::while_(cond_region, body_region);
        self.add_node(instruction, vec![condition])
    }

    pub fn add_block(&mut self, region: RegionId) -> NodeId {
        let instruction = Instruction::block(region);
        self.add_node(instruction, vec![])
    }
}

impl Default for ModuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub trait ValueStore {
    fn resolve_value(&self, value_id: ValueId) -> impl Deref<Target = Value>;
    // Might be a MutexGuard!
    fn resolve_value_mut(&self, value_id: ValueId) -> impl DerefMut<Target = Value>;

    fn set_input(&mut self, func_id: FunctionId, index: usize, value: Value);
    fn set_output(&mut self, func_id: FunctionId, index: usize, value: Value);
    fn set_global(&mut self, global_id: GlobalValueId, value: Value);

    fn pop_input(&mut self, func_id: FunctionId, index: usize) -> Value {
        let mut slot = self.resolve_value_mut(ValueId::Input(func_id, index));
        std::mem::replace(&mut *slot, Value::Empty)
    }

    fn pop_output(&mut self, func_id: FunctionId, index: usize) -> Value {
        let mut slot = self.resolve_value_mut(ValueId::Output(func_id, index));
        std::mem::replace(&mut *slot, Value::Empty)
    }

    fn pop_global(&mut self, global_id: GlobalValueId) -> Value {
        let mut slot = self.resolve_value_mut(ValueId::Global(global_id));
        std::mem::replace(&mut *slot, Value::Empty)
    }
}

#[derive(Debug)]
struct RayonValue(RwLock<Value>);

pub struct RayonValueStore {
    globals: Vec<RayonValue>,
    inputs: Vec<Vec<RayonValue>>,
    outputs: Vec<Vec<RayonValue>>,
}

impl ValueStore for RayonValueStore {
    fn resolve_value(&self, value_id: ValueId) -> impl Deref<Target = Value> {
        match value_id {
            ValueId::Input(func_id, i) => self.inputs[func_id.0][i].0.try_read().unwrap(),
            ValueId::Output(func_id, i) => self.outputs[func_id.0][i].0.try_read().unwrap(),
            ValueId::Global(GlobalValueId(i)) => self.globals[i].0.try_read().unwrap(),
        }
    }

    fn resolve_value_mut(&self, value_id: ValueId) -> impl DerefMut<Target = Value> {
        match value_id {
            ValueId::Input(func_id, i) => self.inputs[func_id.0][i].0.try_write().unwrap(),
            ValueId::Output(func_id, i) => self.outputs[func_id.0][i].0.try_write().unwrap(),
            ValueId::Global(GlobalValueId(i)) => self.globals[i].0.try_write().unwrap(),
        }
    }

    fn set_input(&mut self, func_id: FunctionId, index: usize, value: Value) {
        if func_id.0 >= self.inputs.len() {
            self.inputs.resize_with(func_id.0 + 1, || Vec::new());
        }
        if index >= self.inputs[func_id.0].len() {
            self.inputs[func_id.0].resize_with(index + 1, || RayonValue(RwLock::new(Value::Empty)));
        }
        let mut slot = self.inputs[func_id.0][index].0.write().unwrap();
        *slot = value;
    }

    fn set_output(&mut self, func_id: FunctionId, index: usize, value: Value) {
        if func_id.0 >= self.outputs.len() {
            self.outputs.resize_with(func_id.0 + 1, || Vec::new());
        }
        if index >= self.outputs[func_id.0].len() {
            self.outputs[func_id.0]
                .resize_with(index + 1, || RayonValue(RwLock::new(Value::Empty)));
        }
        let mut slot = self.outputs[func_id.0][index].0.write().unwrap();
        *slot = value;
    }

    fn set_global(&mut self, global_id: GlobalValueId, value: Value) {
        let mut slot = self.globals[global_id.0].0.write().unwrap();
        *slot = value;
    }
}

pub trait ExecutionContext {
    type ValueStore: ValueStore;

    fn run_region(&self, region_id: RegionId, values: &Self::ValueStore) -> Result<()>;
    fn run_region_seq(&self, region_id: RegionId, values: &Self::ValueStore) -> Result<()>;
    fn numba_runtime(&self) -> &Arc<NumbaRuntime>;

    fn create_values(&self) -> Self::ValueStore;
    fn run(&self, func: FunctionId, values: &Self::ValueStore) -> Result<()>;
}

#[derive(Debug)]
pub struct RayonExecutionContext<'a> {
    module: &'a Module,
    // One per NodeId
    missing_counts: Vec<CachePadded<AtomicUsize>>,
    numba_runtime: Arc<NumbaRuntime>,
}

impl<'a> RayonExecutionContext<'a> {
    pub fn new(module: &'a Module, numba_runtime: Arc<NumbaRuntime>) -> Self {
        let missing_counts = (0..module.nodes.len())
            .map(|_| CachePadded::new(AtomicUsize::new(0)))
            .collect();
        Self {
            module,
            missing_counts,
            numba_runtime,
        }
    }

    fn resolve_missing_count(&self, node_id: NodeId) -> &AtomicUsize {
        &self.missing_counts[node_id.0]
    }
}

impl<'a> ExecutionContext for RayonExecutionContext<'a> {
    type ValueStore = RayonValueStore;

    fn run_region(&self, region_id: RegionId, values: &RayonValueStore) -> Result<()> {
        let module = self.module;
        let region = module.resolve_region(region_id);

        // Check if region should be executed sequentially
        if region.sequential {
            return self.run_region_seq(region_id, values);
        }

        let mut ready_nodes: SmallVec<[NodeId; 16]> = SmallVec::new();
        region.nodes.iter().for_each(|&node_id| {
            let node = module.resolve_node(node_id);
            let missing_count = node.predecessors.len();
            let missing_count_loc = self.resolve_missing_count(node_id);
            // TODO check ordering
            missing_count_loc.store(missing_count, std::sync::atomic::Ordering::Relaxed);
            if missing_count == 0 {
                ready_nodes.push(node_id);
            }
        });

        fn run_node(
            ctx: &RayonExecutionContext,
            module: &Module,
            values: &RayonValueStore,
            node_id: NodeId,
        ) -> Result<()> {
            let node = module.resolve_node(node_id);
            node.run(ctx, values)?;

            // Notify successors
            let mut ready_nodes: SmallVec<[NodeId; 16]> = SmallVec::new();
            for &succ_id in node.successors.iter() {
                let succ_missing_count = ctx.resolve_missing_count(succ_id);
                let prev_count =
                    succ_missing_count.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                assert!(prev_count > 0);
                if prev_count == 1 {
                    // Now ready
                    let node_id = succ_id;
                    ready_nodes.push(node_id);
                }
            }

            ready_nodes
                .into_par_iter()
                .try_for_each(|&node_id| run_node(ctx, module, values, node_id))?;
            Ok(())
        }

        ready_nodes
            .into_par_iter()
            .try_for_each(|&node_id| run_node(self, module, values, node_id))?;
        Ok(())
    }

    fn run_region_seq(&self, region_id: RegionId, values: &RayonValueStore) -> Result<()> {
        let module = self.module;
        let region = module.resolve_region(region_id);

        // Sequential execution - just iterate through nodes in order
        for &node_id in &region.nodes {
            let node = module.resolve_node(node_id);
            node.run(self, values)?;
        }
        Ok(())
    }

    fn numba_runtime(&self) -> &Arc<NumbaRuntime> {
        &self.numba_runtime
    }

    fn create_values(&self) -> Self::ValueStore {
        let module = self.module;
        // Count the maximum global value ID used across all nodes
        let mut max_global_id = 0;
        for node in &module.nodes {
            for &arg in &node.args {
                if let ValueId::Global(GlobalValueId(id)) = arg {
                    max_global_id = max_global_id.max(id + 1);
                }
            }
        }

        let mut globals = Vec::with_capacity(max_global_id);
        for _ in 0..max_global_id {
            globals.push(RayonValue(RwLock::new(Value::Empty)));
        }

        RayonValueStore {
            globals,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    fn run(&self, function_id: FunctionId, values: &Self::ValueStore) -> Result<()> {
        let entry = self.module.resolve_function(function_id).entry;
        self.run_region(entry, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{DType, Order};
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_builder_pattern() {
        // Create the module builder inside an Arc<Mutex<_>>
        let module_builder = Arc::new(Mutex::new(ModuleBuilder::new()));

        // Add some global variables
        let temp_array_id = {
            let mut builder = module_builder.lock().unwrap();
            builder.add_global_tensor(
                Some("temp_array".to_string()),
                TensorType {
                    dtype: DType::F32,
                    order: Order::RowMajor,
                    shape: Some([None, None].as_slice().into()),
                },
            )
        };

        let condition_id = {
            let mut builder = module_builder.lock().unwrap();
            builder.add_global_tensor(
                Some("condition".to_string()),
                TensorType {
                    dtype: DType::Bool,
                    order: Order::RowMajor,
                    shape: Some([].as_slice().into()),
                },
            )
        };

        // Create a function
        let function_builder = {
            let mut builder = module_builder.lock().unwrap();
            builder.create_function(
                "main".to_string(),
                vec![
                    Variable::new(
                        ValueType::Tensor(TensorType {
                            dtype: DType::F32,
                            order: Order::RowMajor,
                            shape: Some([None, None].as_slice().into()),
                        }),
                        None,
                    ),
                    Variable::new(
                        ValueType::Tensor(TensorType {
                            dtype: DType::I64,
                            order: Order::RowMajor,
                            shape: Some([None].as_slice().into()),
                        }),
                        None,
                    ),
                ],
                vec![Variable::new(
                    ValueType::Tensor(TensorType {
                        dtype: DType::F32,
                        order: Order::RowMajor,
                        shape: Some([None, None].as_slice().into()),
                    }),
                    None,
                )],
            )
        };

        // Create the main region
        let mut region_builder = function_builder.create_region(module_builder.clone());

        // Add an allocation instruction
        let alloc_node = region_builder.add_alloc(
            temp_array_id,
            function_builder.input(1).unwrap(), // shape input
            TensorType {
                dtype: DType::F32,
                order: Order::RowMajor,
                shape: Some([None, None].as_slice().into()),
            },
        );

        // Create subregions for if branches
        let then_region_id = {
            let then_builder = region_builder.create_subregion();
            // Add some node to the then branch (placeholder)
            let (_, region_id) = then_builder.finish();
            region_id
        };

        let else_region_id = {
            let else_builder = region_builder.create_subregion();
            // Add some node to the else branch (placeholder)
            let (_, region_id) = else_builder.finish();
            region_id
        };

        // Add an if instruction
        let if_node = region_builder.add_if(condition_id, then_region_id, else_region_id);

        // Add dependency
        region_builder.add_dependency(alloc_node, if_node);

        // Finish the region
        let (function_builder, main_region_id) = region_builder.finish();
        let function_builder = function_builder.unwrap();

        // Finish the function
        let _function_id = function_builder.finish(module_builder.clone(), main_region_id);

        // Build the final module
        let module = {
            let builder =
                std::mem::replace(&mut *module_builder.lock().unwrap(), ModuleBuilder::new());
            builder.build()
        };

        // Test the display output
        println!("{}", module);

        // Verify structure
        assert_eq!(module.num_functions(), 1);
        assert_eq!(module.num_variables(), 2);
        assert!(module.num_regions() >= 3); // main + then + else
    }

    #[test]
    fn test_complex_builder_example() {
        use crate::ffi::ExternalArrayFunc;

        // Create the module builder inside an Arc<Mutex<_>>
        let module_builder = Arc::new(Mutex::new(ModuleBuilder::new()));

        // Add global variables
        let input_array_id = {
            let mut builder = module_builder.lock().unwrap();
            builder.add_global_tensor(
                Some("input_data".to_string()),
                TensorType {
                    dtype: DType::F32,
                    order: Order::RowMajor,
                    shape: Some([None, None].as_slice().into()),
                },
            )
        };

        let output_array_id = {
            let mut builder = module_builder.lock().unwrap();
            builder.add_global_tensor(
                Some("output_data".to_string()),
                TensorType {
                    dtype: DType::F32,
                    order: Order::RowMajor,
                    shape: Some([None, None].as_slice().into()),
                },
            )
        };

        let condition_id = {
            let mut builder = module_builder.lock().unwrap();
            builder.add_global_tensor(
                Some("should_transform".to_string()),
                TensorType {
                    dtype: DType::Bool,
                    order: Order::RowMajor,
                    shape: Some([].as_slice().into()),
                },
            )
        };

        // Create a function with inputs and outputs
        let function_builder = {
            let mut builder = module_builder.lock().unwrap();
            builder.create_function(
                "process_data".to_string(),
                vec![
                    Variable::new(
                        ValueType::Tensor(TensorType {
                            dtype: DType::F32,
                            order: Order::RowMajor,
                            shape: Some([None, None].as_slice().into()),
                        }),
                        Some("input".to_string()),
                    ),
                    Variable::new(
                        ValueType::Tensor(TensorType {
                            dtype: DType::I64,
                            order: Order::RowMajor,
                            shape: Some([None].as_slice().into()),
                        }),
                        Some("shape".to_string()),
                    ),
                ],
                vec![Variable::new(
                    ValueType::Tensor(TensorType {
                        dtype: DType::F32,
                        order: Order::RowMajor,
                        shape: Some([None, None].as_slice().into()),
                    }),
                    Some("result".to_string()),
                )],
            )
        };

        // Create the main graph
        let mut main_region = function_builder.create_region(module_builder.clone());

        // Add allocation for output
        main_region.add_alloc(
            output_array_id,
            function_builder.input(1).unwrap(), // shape input
            TensorType {
                dtype: DType::F32,
                order: Order::RowMajor,
                shape: Some([None, None].as_slice().into()),
            },
        );

        // Create dummy external functions for testing
        extern "C" fn compute_condition_fn(
            _size: usize,
            _buffers: *mut *mut std::ffi::c_void,
            _meminfos: *mut *mut crate::ffi::NRT_MemInfo,
            _nbytes: *mut usize,
            _ranks: *const usize,
            _shapes: *mut usize,
            _strides: *mut usize,
            _dtypes: *mut u8,
            _modified: *mut u8,
        ) -> i64 {
            0 // Success
        }

        extern "C" fn operation_a_fn(
            _size: usize,
            _buffers: *mut *mut std::ffi::c_void,
            _meminfos: *mut *mut crate::ffi::NRT_MemInfo,
            _nbytes: *mut usize,
            _ranks: *const usize,
            _shapes: *mut usize,
            _strides: *mut usize,
            _dtypes: *mut u8,
            _modified: *mut u8,
        ) -> i64 {
            0 // Success
        }

        extern "C" fn operation_b_fn(
            _size: usize,
            _buffers: *mut *mut std::ffi::c_void,
            _meminfos: *mut *mut crate::ffi::NRT_MemInfo,
            _nbytes: *mut usize,
            _ranks: *const usize,
            _shapes: *mut usize,
            _strides: *mut usize,
            _dtypes: *mut u8,
            _modified: *mut u8,
        ) -> i64 {
            0 // Success
        }

        extern "C" fn copy_result_fn(
            _size: usize,
            _buffers: *mut *mut std::ffi::c_void,
            _meminfos: *mut *mut crate::ffi::NRT_MemInfo,
            _nbytes: *mut usize,
            _ranks: *const usize,
            _shapes: *mut usize,
            _strides: *mut usize,
            _dtypes: *mut u8,
            _modified: *mut u8,
        ) -> i64 {
            0 // Success
        }

        // Add external call to compute condition
        let condition_node = main_region.add_external_call(
            ExternalArrayFunc::new(compute_condition_fn),
            None,
            vec![function_builder.input(0).unwrap(), condition_id],
            vec![false, true], // input is immutable, condition is mutable
            "compute_condition".to_string(),
        );

        let tensor_type = TensorType {
            dtype: DType::F32,
            order: Order::RowMajor,
            shape: Some([None, None].as_slice().into()),
        };

        let temp_array_id = {
            let mut builder = module_builder.lock().unwrap();
            builder.add_global_tensor(Some("temp_array".to_string()), tensor_type.clone())
        };

        // Add an allocation instruction
        let alloc_node = main_region.add_alloc(
            temp_array_id,
            function_builder.input(1).unwrap(),
            tensor_type,
        );

        // Create subregions for if branches
        let then_region_id = {
            let mut then_builder = main_region.create_subregion();
            then_builder.add_external_call(
                ExternalArrayFunc::new(operation_a_fn),
                None,
                vec![function_builder.input(0).unwrap(), input_array_id],
                vec![false, true], // input immutable, temp mutable
                "operation_a".to_string(),
            );
            let (_, region_id) = then_builder.finish();
            region_id
        };

        let else_region_id = {
            let mut else_builder = main_region.create_subregion();
            else_builder.add_external_call(
                ExternalArrayFunc::new(operation_b_fn),
                None,
                vec![function_builder.input(0).unwrap(), input_array_id],
                vec![false, true], // input immutable, temp mutable
                "operation_b".to_string(),
            );
            let (_, region_id) = else_builder.finish();
            region_id
        };

        // Add conditional execution
        let if_node = main_region.add_if(condition_id, then_region_id, else_region_id);

        // Add final copy operation
        let copy_node = main_region.add_external_call(
            ExternalArrayFunc::new(copy_result_fn),
            None,
            vec![input_array_id, function_builder.output(0).unwrap()],
            vec![false, true], // src immutable, dst mutable
            "copy_result".to_string(),
        );

        // Set up dependencies
        main_region.add_dependency(alloc_node, condition_node);
        main_region.add_dependency(condition_node, if_node);
        main_region.add_dependency(if_node, copy_node);

        // Finish the region and function
        let (function_builder, main_region_id) = main_region.finish();
        let function_builder = function_builder.unwrap();
        let _function_id = function_builder.finish(module_builder.clone(), main_region_id);

        // Build the final module
        let module = {
            let builder =
                std::mem::replace(&mut *module_builder.lock().unwrap(), ModuleBuilder::new());
            builder.build()
        };

        // Print the generated code
        println!("Generated Module:");
        println!("{}", module);

        // Verify structure
        assert_eq!(module.num_functions(), 1);
        assert_eq!(module.num_variables(), 3); // input_data, output_data, should_transform
        assert!(module.num_regions() >= 3); // main + then + else branches
    }
}
