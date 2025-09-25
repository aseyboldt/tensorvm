use std::fmt::{Debug, Display};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tensorvm::{
    DType, DynInstruction, Error, ExecutionContext, Instruction, ModuleBuilder, NumbaRuntime,
    Order, RayonExecutionContext, Tensor, TensorType, Value, ValueId, ValueStore, ValueType,
    Variable,
};

const N: usize = 4;

// Simple addition operation that implements DynInstruction
#[derive(Debug)]
struct AddOperation;

impl Display for AddOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Add")
    }
}

impl DynInstruction for AddOperation {
    fn run_dyn(&self, inputs: &[&Value], outputs: &mut [&mut Value]) -> Result<(), Error> {
        if inputs.len() != 2 {
            return Err(Error::ValueError {
                message: format!("Add expects 2 inputs, got {}", inputs.len()),
            });
        }
        if outputs.len() != 1 {
            return Err(Error::ValueError {
                message: format!("Add expects 1 output, got {}", outputs.len()),
            });
        }

        // Get input tensors
        let a = match &inputs[0] {
            Value::Tensor(t) => t,
            Value::Empty => {
                return Err(Error::ValueError {
                    message: "First input is empty".to_string(),
                });
            }
        };

        let b = match &inputs[1] {
            Value::Tensor(t) => t,
            Value::Empty => {
                return Err(Error::ValueError {
                    message: "Second input is empty".to_string(),
                });
            }
        };

        let c = match &mut outputs[0] {
            Value::Tensor(t) => t,
            Value::Empty => {
                return Err(Error::ValueError {
                    message: "Second input is empty".to_string(),
                });
            }
        };

        let a = a.as_array_view::<f32>()?.unwrap();
        let b = b.as_array_view::<f32>()?.unwrap();
        let mut c = c.as_array_view_mut::<f32>()?.unwrap();

        c.assign(&(&a + &b));
        c.iter_mut().for_each(|x| {
            *x = x.sin();
        });

        Ok(())
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "Add"
    }
}

// Multiplication operation
#[derive(Debug)]
struct MulOperation;

impl Display for MulOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mul")
    }
}

impl DynInstruction for MulOperation {
    fn run_dyn(&self, inputs: &[&Value], outputs: &mut [&mut Value]) -> Result<(), Error> {
        if inputs.len() != 2 {
            return Err(Error::ValueError {
                message: format!("Mul expects 2 inputs, got {}", inputs.len()),
            });
        }
        if outputs.len() != 1 {
            return Err(Error::ValueError {
                message: format!("Mul expects 1 output, got {}", outputs.len()),
            });
        }

        let a = match &inputs[0] {
            Value::Tensor(t) => t,
            Value::Empty => {
                return Err(Error::ValueError {
                    message: "First input is empty".to_string(),
                });
            }
        };

        let b = match &inputs[1] {
            Value::Tensor(t) => t,
            Value::Empty => {
                return Err(Error::ValueError {
                    message: "Second input is empty".to_string(),
                });
            }
        };

        let c = match &mut outputs[0] {
            Value::Tensor(t) => t,
            Value::Empty => {
                return Err(Error::ValueError {
                    message: "Second input is empty".to_string(),
                });
            }
        };

        let a = a.as_array_view::<f32>()?.unwrap();
        let b = b.as_array_view::<f32>()?.unwrap();
        let mut c = c.as_array_view_mut::<f32>()?.unwrap();

        c.assign(&(&a * &b));

        Ok(())
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "Mul"
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating tensor operations module...");

    // Create module builder
    let module_builder = Arc::new(Mutex::new(ModuleBuilder::new()));

    // Define tensor types
    let f32_tensor = TensorType {
        dtype: DType::F32,
        order: Order::RowMajor,
        shape: Some([None, None].as_slice().into()),
    };

    let shape_tensor = TensorType {
        dtype: DType::I64,
        order: Order::RowMajor,
        shape: Some([Some(2)].as_slice().into()), // Fixed shape: [2]
    };

    // Add global variables for intermediate results
    let temp1_id = {
        let mut builder = module_builder.lock().unwrap();
        builder.add_global_tensor(Some("temp1".to_string()), f32_tensor.clone())
    };

    let temp2_id = {
        let mut builder = module_builder.lock().unwrap();
        builder.add_global_tensor(Some("temp2".to_string()), f32_tensor.clone())
    };

    let condition_id = {
        let mut builder = module_builder.lock().unwrap();
        builder.add_global_tensor(
            Some("condition".to_string()),
            TensorType {
                dtype: DType::I32,
                order: Order::RowMajor,
                shape: Some([].as_slice().into()),
            },
        )
    };

    // Create main function
    let function_builder = {
        let mut builder = module_builder.lock().unwrap();
        builder.create_function(
            "compute".to_string(),
            vec![
                Variable::new(
                    ValueType::Tensor(f32_tensor.clone()),
                    Some("input_a".to_string()),
                ),
                Variable::new(
                    ValueType::Tensor(f32_tensor.clone()),
                    Some("input_b".to_string()),
                ),
                Variable::new(
                    ValueType::Tensor(shape_tensor.clone()),
                    Some("shape".to_string()),
                ),
            ],
            vec![Variable::new(
                ValueType::Tensor(f32_tensor.clone()),
                Some("result".to_string()),
            )],
        )
    };

    // Build the computation graph
    let mut main_region = function_builder
        .create_region(module_builder.clone())
        .parallel();

    // Allocate temporary tensors
    let alloc1_node = main_region.add_alloc(
        temp1_id,
        function_builder.input(2).unwrap(), // shape input
        f32_tensor.clone(),
    );

    let alloc2_node = main_region.add_alloc(
        temp2_id,
        function_builder.input(2).unwrap(), // shape input
        f32_tensor.clone(),
    );

    // Add operations using dyn_instruction
    let add_node = main_region.add_node(
        Instruction::dyn_instruction(Box::new(AddOperation)),
        vec![
            function_builder.input(0).unwrap(), // input_a
            function_builder.input(1).unwrap(), // input_b
            temp1_id,                           // output
        ],
    );

    let mul_node = main_region.add_node(
        Instruction::dyn_instruction(Box::new(MulOperation)),
        vec![
            temp1_id,                           // result from add
            function_builder.input(0).unwrap(), // input_a again
            temp2_id,                           // output
        ],
    );

    // Create conditional branches - mark small branches as sequential
    let then_region_id = {
        let mut then_builder = main_region.create_subregion().sequential();
        // In then branch, copy temp2 to output using add with zero
        then_builder.add_node(
            Instruction::dyn_instruction(Box::new(AddOperation)),
            vec![
                temp2_id,
                temp2_id, // Add to itself (simplified copy)
                function_builder.output(0).unwrap(),
            ],
        );
        let (_, region_id) = then_builder.finish();
        region_id
    };

    let else_region_id = {
        let mut else_builder = main_region.create_subregion().sequential();
        // In else branch, copy temp1 to output
        else_builder.add_node(
            Instruction::dyn_instruction(Box::new(AddOperation)),
            vec![
                temp1_id,
                temp1_id, // Add to itself (simplified copy)
                function_builder.output(0).unwrap(),
            ],
        );
        let (_, region_id) = else_builder.finish();
        region_id
    };

    // Add conditional execution
    let if_node = main_region.add_if(condition_id, then_region_id, else_region_id);

    // Set up dependencies
    main_region.add_dependency(alloc1_node, add_node);
    main_region.add_dependency(alloc2_node, mul_node);
    main_region.add_dependency(add_node, mul_node);
    main_region.add_dependency(mul_node, if_node);

    // Finish building
    let (function_builder, main_region_id) = main_region.finish();
    let function_builder = function_builder.unwrap();
    let function_id = function_builder.finish(module_builder.clone(), main_region_id);

    // Build the module
    let module = {
        let builder = std::mem::replace(&mut *module_builder.lock().unwrap(), ModuleBuilder::new());
        builder.build()
    };

    // Display the module IR
    println!("Generated Module IR:");
    println!("{}", module);

    // Describe what the computation does
    println!("\nComputation Description:");
    println!("1. Allocate two temporary tensors (temp1, temp2)");
    println!("2. Add input_a + input_b -> temp1");
    println!("3. Multiply temp1 * input_a -> temp2");
    println!("4. Conditionally choose result:");
    println!("   - If condition is true: result = temp2 (processed result)");
    println!("   - If condition is false: result = temp1 (intermediate result)");
    println!("5. Sequential block: result = result + result (demonstrates Block instruction)");

    // Create execution context and run the function
    println!("\nExecuting the function...");

    let numba_runtime = Arc::new(NumbaRuntime::make_panic_runtime());
    let ctx = RayonExecutionContext::new(&module, numba_runtime);
    let mut values = ctx.create_values();

    // Set up input values
    let mut input_a = Tensor::new(DType::F32, &[N, 3], Order::RowMajor);
    let mut input_b = Tensor::new(DType::F32, &[N, 3], Order::RowMajor);
    let mut shape_tensor = Tensor::new(DType::I64, &[2], Order::RowMajor);
    let condition_tensor = Tensor::new(DType::I32, &[], Order::RowMajor);

    let mut x = input_a.as_array_view_mut::<f32>()?.unwrap();
    x.fill(1.0);

    let mut y = input_b.as_array_view_mut::<f32>()?.unwrap();
    y.fill(2.0);

    let mut shape_view = shape_tensor.as_array_view_mut::<i64>()?.unwrap();
    shape_view[0] = N as i64;
    shape_view[1] = 3;

    println!("Setting up input values...");
    values.set_input(function_id, 0, Value::Tensor(input_a));
    values.set_input(function_id, 1, Value::Tensor(input_b));
    values.set_input(function_id, 2, Value::Tensor(shape_tensor));

    // Set the condition global variable
    if let ValueId::Global(global_id) = condition_id {
        values.set_global(global_id, Value::Tensor(condition_tensor));
    }

    // Allocate output tensor
    let output_tensor = Tensor::new(DType::F32, &[N, 3], Order::RowMajor);
    values.set_output(function_id, 0, Value::Tensor(output_tensor));

    // Run the function once to verify it works
    println!("Executing computation graph...");
    match ctx.run(function_id, &values) {
        Ok(()) => {
            println!("✓ Function executed successfully!");
            println!("✓ All operations completed without errors");
            println!("✓ Computation graph: input_a + input_b -> temp1, temp1 * input_a -> temp2");
            println!("✓ Conditional execution: if condition then temp2 else temp1 -> result");
            println!("✓ Sequential block execution: result = result + result");
        }
        Err(e) => {
            println!("✗ Execution failed: {}", e);
            return Ok(());
        }
    }

    // Performance timing
    println!("\nPerformance Timing:");
    println!("Running function 100 times...");

    let mut execution_times = Vec::new();

    for i in 0..10_000 {
        let start = Instant::now();
        match ctx.run(function_id, &values) {
            Ok(()) => {
                let duration = start.elapsed();
                execution_times.push(duration);

                if (i + 1) % 1000 == 0 {
                    println!("Completed {} runs", i + 1);
                }
            }
            Err(e) => {
                println!("✗ Execution {} failed: {}", i + 1, e);
                break;
            }
        }
    }

    // Print timing statistics
    if !execution_times.is_empty() {
        println!("\nTiming Results ({} runs):", execution_times.len());

        // Calculate statistics
        let total_time: std::time::Duration = execution_times.iter().sum();
        let avg_time = total_time / execution_times.len() as u32;

        let mut sorted_times = execution_times.clone();
        sorted_times.sort();

        let min_time = sorted_times.first().unwrap();
        let max_time = sorted_times.last().unwrap();
        let median_time = sorted_times[sorted_times.len() / 2];

        println!("  Average: {:?}", avg_time);
        println!("  Median:  {:?}", median_time);
        println!("  Min:     {:?}", min_time);
        println!("  Max:     {:?}", max_time);
        println!("  Total:   {:?}", total_time);

        // Print first 10 individual timings
        println!("\nFirst 10 execution times:");
        for (i, time) in execution_times.iter().take(10).enumerate() {
            println!("  Run {}: {:?}", i + 1, time);
        }

        if execution_times.len() > 10 {
            println!("  ... ({} more runs)", execution_times.len() - 10);
        }
    }

    // Verify structure
    println!("\nModule Statistics:");
    println!("Functions: {}", module.num_functions());
    println!("Global variables: {}", module.num_variables());
    println!("Regions: {}", module.num_regions());
    println!("Nodes: {}", module.num_nodes());

    println!("Example demonstrates:");
    println!("- Creating custom DynInstruction operations (Add, Mul)");
    println!("- Building computation graphs with control flow");
    println!("- Managing memory allocation and tensor operations");
    println!("- Sequential vs parallel region execution with Block instruction");
    println!("- Generating readable IR output for debugging");

    Ok(())
}
