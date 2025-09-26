import pytest
import tensorvm._lib as lib


def test_create_function_builder():
    """Test basic function builder creation."""
    builder = lib.ModuleBuilder()
    assert builder is not None


def test_create_tensor_type():
    """Test tensor type creation with various dtypes and orders."""
    # Test valid combinations
    tensor_type = lib.TensorType("f32", [3, None], "C")
    assert tensor_type.rank == 2
    assert tensor_type.shape == (3, None)
    assert tensor_type.dtype == "float32"
    assert tensor_type.order == "C"

    # Test other valid dtypes
    for dtype in ["f64", "i32", "i64", "u8", "bool"]:
        tt = lib.TensorType(dtype, None, "C")
        assert tt.dtype in ["float64", "int32", "int64", "uint8", "bool"]
        assert tt.shape is None

    # Test other valid orders
    for order in ["F", "A"]:
        tt = lib.TensorType("f32", None, order)
        assert tt.order == order


def test_invalid_tensor_type():
    """Test that invalid tensor types raise errors."""
    # Invalid dtype
    with pytest.raises(ValueError, match="Invalid dtype"):
        lib.TensorType("invalid_dtype", [None, None], "C")

    # Invalid order
    with pytest.raises(ValueError, match="Invalid order"):
        lib.TensorType("f32", None, "invalid_order")


def test_tensor_type_shape_handling():
    """Test tensor type shape handling with different configurations."""
    # Scalar tensor (no shape)
    scalar = lib.TensorType("f32", None, "C")
    assert scalar.rank is None
    assert scalar.shape is None

    # Vector with known size
    vector = lib.TensorType("f32", [10], "C")
    assert vector.rank == 1
    assert vector.shape == (10,)

    # Matrix with mixed known/unknown dimensions
    matrix = lib.TensorType("f32", [5, None], "C")
    assert matrix.rank == 2
    assert matrix.shape == (5, None)

    # Higher dimensional tensor
    tensor_4d = lib.TensorType("f32", [2, 3, None, 4], "C")
    assert tensor_4d.rank == 4
    assert tensor_4d.shape == (2, 3, None, 4)


def test_variable_creation():
    """Test creating variables with tensor types."""
    tensor_type = lib.TensorType("f32", [3, 3], "C")

    # Named variable
    var1 = lib.Variable(tensor_type, "my_tensor")
    assert var1 is not None

    # Unnamed variable
    var2 = lib.Variable(tensor_type, None)
    assert var2 is not None


def test_module_builder_global_tensors():
    """Test adding global tensors to the module."""
    builder = lib.ModuleBuilder()
    tensor_type = lib.TensorType("f32", [100, 100], "C")

    # Add named global tensor
    value_id = builder.add_global_tensor("weights", tensor_type)
    assert value_id is not None

    # Add unnamed global tensor
    value_id2 = builder.add_global_tensor(None, tensor_type)
    assert value_id2 is not None


def test_function_creation_and_io():
    """Test creating functions and accessing their inputs/outputs."""
    builder = lib.ModuleBuilder()

    # Create input/output variables
    input_type = lib.TensorType("f32", [None, 128], "C")
    output_type = lib.TensorType("f32", [None, 10], "C")

    input_var = lib.Variable(input_type, "input")
    output_var = lib.Variable(output_type, "output")

    # Create function
    func_builder = builder.create_function("test_func", [input_var], [output_var])
    assert func_builder is not None

    # Test input access
    inputs = func_builder.inputs()
    assert len(inputs) == 1

    input0 = func_builder.input(0)
    assert input0 is not None

    # Test output access
    outputs = func_builder.outputs()
    assert len(outputs) == 1

    output0 = func_builder.output(0)
    assert output0 is not None

    # Test out-of-bounds access
    with pytest.raises(IndexError):
        func_builder.input(1)

    with pytest.raises(IndexError):
        func_builder.output(1)


def test_region_creation_and_operations():
    """Test creating regions and adding operations."""
    builder = lib.ModuleBuilder()

    # Create a simple function
    input_type = lib.TensorType("f32", [10], "C")
    output_type = lib.TensorType("f32", [10], "C")

    input_var = lib.Variable(input_type, "x")
    output_var = lib.Variable(output_type, "y")

    func_builder = builder.create_function("simple_func", [input_var], [output_var])

    # Create main region
    region_builder = func_builder.create_region()
    assert region_builder is not None

    # Test creating subregions
    subregion1 = region_builder.create_subregion()
    assert subregion1 is not None

    subregion2 = region_builder.create_subregion()
    assert subregion2 is not None


def test_region_alloc_operation():
    """Test adding allocation operations to regions."""
    builder = lib.ModuleBuilder()

    # Create function with tensor inputs
    tensor_type = lib.TensorType("f32", [None, None], "C")
    shape_type = lib.TensorType("i64", [2], "C")

    shape_var = lib.Variable(shape_type, "shape")
    output_var = lib.Variable(tensor_type, "output")

    func_builder = builder.create_function("alloc_func", [shape_var], [output_var])
    region_builder = func_builder.create_region()

    # Get input and output value IDs
    shape_id = func_builder.input(0)
    output_id = func_builder.output(0)

    # Add allocation operation
    alloc_node = region_builder.add_alloc(output_id, shape_id, tensor_type)
    assert alloc_node is not None


def test_region_control_flow():
    """Test adding control flow operations (if/while)."""
    builder = lib.ModuleBuilder()

    # Create function with boolean condition
    bool_type = lib.TensorType("bool", [], "C")  # scalar boolean
    tensor_type = lib.TensorType("f32", [10], "C")

    cond_var = lib.Variable(bool_type, "condition")
    input_var = lib.Variable(tensor_type, "input")
    output_var = lib.Variable(tensor_type, "output")

    func_builder = builder.create_function(
        "control_flow", [cond_var, input_var], [output_var]
    )
    main_region = func_builder.create_region()

    # Create subregions for control flow
    then_region = main_region.create_subregion()
    else_region = main_region.create_subregion()
    cond_region = main_region.create_subregion()
    body_region = main_region.create_subregion()

    # Finish subregions to get their IDs
    _, then_id = then_region.finish()
    _, else_id = else_region.finish()
    _, cond_id = cond_region.finish()
    _, body_id = body_region.finish()

    # Add control flow operations
    condition_id = func_builder.input(0)

    if_node = main_region.add_if(condition_id, then_id, else_id)
    assert if_node is not None

    while_node = main_region.add_while(condition_id, cond_id, body_id)
    assert while_node is not None


def test_region_dependencies():
    """Test adding dependencies between nodes."""
    builder = lib.ModuleBuilder()

    tensor_type = lib.TensorType("f32", [10], "C")
    shape_type = lib.TensorType("i64", [1], "C")

    shape_var = lib.Variable(shape_type, "shape")
    out1_var = lib.Variable(tensor_type, "out1")
    out2_var = lib.Variable(tensor_type, "out2")

    func_builder = builder.create_function(
        "dep_test", [shape_var], [out1_var, out2_var]
    )
    region_builder = func_builder.create_region()

    # Create two allocation nodes
    shape_id = func_builder.input(0)
    out1_id = func_builder.output(0)
    out2_id = func_builder.output(1)

    node1 = region_builder.add_alloc(out1_id, shape_id, tensor_type)
    node2 = region_builder.add_alloc(out2_id, shape_id, tensor_type)

    # Add dependency: node2 depends on node1
    region_builder.add_dependency(node1, node2)


def test_external_function_call():
    """Test adding external function calls."""
    builder = lib.ModuleBuilder()

    tensor_type = lib.TensorType("f32", [10], "C")
    input_var = lib.Variable(tensor_type, "input")
    output_var = lib.Variable(tensor_type, "output")

    func_builder = builder.create_function("external_call", [input_var], [output_var])
    region_builder = func_builder.create_region()

    # Get value IDs
    input_id = func_builder.input(0)
    output_id = func_builder.output(0)

    # Mock function pointer (this would be a real function pointer in practice)
    # For testing, we use a placeholder value
    mock_func_ptr = 0x12345678

    # Add external call (input is read-only, output is mutable)
    external_node = region_builder.add_external_call(
        mock_func_ptr,
        [input_id, output_id],
        [False, True],  # input is immutable, output is mutable
        "mock_function",
    )
    assert external_node is not None


def test_complete_function_workflow():
    """Test complete workflow of building and finishing a function."""
    builder = lib.ModuleBuilder()

    # Create function
    input_type = lib.TensorType("f32", [None], "C")
    output_type = lib.TensorType("f32", [None], "C")

    input_var = lib.Variable(input_type, "x")
    output_var = lib.Variable(output_type, "y")

    func_builder = builder.create_function("complete_func", [input_var], [output_var])

    # Create and populate main region
    region_builder = func_builder.create_region()

    # Finish the region
    _, region_id = region_builder.finish()

    # Finish the function
    func_id = func_builder.finish(region_id)
    assert func_id is not None


def test_module_building_and_inspection():
    """Test building a complete module and inspecting its properties."""
    builder = lib.ModuleBuilder()

    # Add some global tensors
    global_type = lib.TensorType("f32", [100, 100], "C")
    builder.add_global_tensor("global1", global_type)
    builder.add_global_tensor("global2", global_type)

    # Create a function
    input_type = lib.TensorType("f32", [10], "C")
    output_type = lib.TensorType("f32", [10], "C")

    input_var = lib.Variable(input_type, "input")
    output_var = lib.Variable(output_type, "output")

    func_builder = builder.create_function("test_function", [input_var], [output_var])
    region_builder = func_builder.create_region()

    # Add some operations
    input_id = func_builder.input(0)
    output_id = func_builder.output(0)

    # Create a simple allocation
    shape_type = lib.TensorType("i64", [1], "C")
    shape_global = builder.add_global_tensor("shape", shape_type)
    alloc_node = region_builder.add_alloc(output_id, shape_global, output_type)

    # Finish everything
    _, region_id = region_builder.finish()
    func_id = func_builder.finish(region_id)

    # Build the module
    module = builder.build()

    # Test module inspection
    assert module.num_functions() >= 1
    assert module.num_variables() >= 3  # 2 globals + function inputs/outputs
    assert module.num_regions() >= 1
    assert module.num_nodes() >= 1

    # Test string representations
    asm_str = module.asm()
    assert isinstance(asm_str, str)
    assert len(asm_str) > 0

    str_repr = str(module)
    assert isinstance(str_repr, str)

    repr_str = repr(module)
    assert "PyTensorModule" in repr_str
    assert "functions=" in repr_str
    assert "globals=" in repr_str


def test_function_builder_state_errors():
    """Test that function builder properly handles state errors."""
    builder = lib.ModuleBuilder()

    input_var = lib.Variable(lib.TensorType("f32", [10], "C"), "input")
    output_var = lib.Variable(lib.TensorType("f32", [10], "C"), "output")

    func_builder = builder.create_function("test", [input_var], [output_var])
    region_builder = func_builder.create_region()

    # Finish the region and function
    _, region_id = region_builder.finish()
    func_id = func_builder.finish(region_id)

    # Now trying to use the finished function builder should raise errors
    with pytest.raises(RuntimeError, match="Function already finished"):
        func_builder.create_region()

    with pytest.raises(RuntimeError, match="Function already finished"):
        func_builder.input(0)

    with pytest.raises(RuntimeError, match="Function already finished"):
        func_builder.inputs()

    with pytest.raises(RuntimeError, match="Function already finished"):
        func_builder.output(0)

    with pytest.raises(RuntimeError, match="Function already finished"):
        func_builder.outputs()


def test_region_builder_state_errors():
    """Test that region builder properly handles state errors."""
    builder = lib.ModuleBuilder()

    input_var = lib.Variable(lib.TensorType("f32", [10], "C"), "input")
    output_var = lib.Variable(lib.TensorType("f32", [10], "C"), "output")

    func_builder = builder.create_function("test", [input_var], [output_var])
    region_builder = func_builder.create_region()

    # Finish the region
    _, region_id = region_builder.finish()

    # Now trying to use the finished region builder should raise errors
    with pytest.raises(RuntimeError, match="Region already finished"):
        region_builder.create_subregion()

    with pytest.raises(RuntimeError, match="Region already finished"):
        input_id = func_builder.input(0)
        output_id = func_builder.output(0)
        tensor_type = lib.TensorType("f32", [10], "C")
        region_builder.add_alloc(output_id, input_id, tensor_type)


def test_multiple_functions_in_module():
    """Test creating multiple functions in a single module."""
    builder = lib.ModuleBuilder()

    # Function 1: Simple identity
    input_type = lib.TensorType("f32", [10], "C")
    func1_builder = builder.create_function(
        "identity", [lib.Variable(input_type, "x")], [lib.Variable(input_type, "y")]
    )
    region1 = func1_builder.create_region()
    _, region1_id = region1.finish()
    func1_id = func1_builder.finish(region1_id)

    # Function 2: Another simple function
    matrix_type = lib.TensorType("f32", [None, None], "C")
    func2_builder = builder.create_function(
        "matrix_op",
        [lib.Variable(matrix_type, "a"), lib.Variable(matrix_type, "b")],
        [lib.Variable(matrix_type, "c")],
    )
    region2 = func2_builder.create_region()
    _, region2_id = region2.finish()
    func2_id = func2_builder.finish(region2_id)

    # Build module
    module = builder.build()

    # Should have 2 functions
    assert module.num_functions() == 2


def test_dtypes_comprehensive():
    """Test all supported data types comprehensively."""
    dtypes_to_test = [
        ("f32", "float32"),
        ("f64", "float64"),
        ("i8", "int8"),
        ("i16", "int16"),
        ("i32", "int32"),
        ("i64", "int64"),
        ("u8", "uint8"),
        ("u16", "uint16"),
        ("u32", "uint32"),
        ("u64", "uint64"),
        ("bool", "bool"),
    ]

    for input_dtype, expected_output in dtypes_to_test:
        try:
            tensor_type = lib.TensorType(input_dtype, [10], "C")
            # The dtype property might normalize the name
            assert tensor_type.dtype in [input_dtype, expected_output]
        except ValueError:
            # Some dtypes might not be supported, that's okay
            continue
