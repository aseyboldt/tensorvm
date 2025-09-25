# ffi_numba_wrap_overload.py
import numpy as np
import numba
from numba import njit, cfunc, carray, types, literal_unroll
from numba.extending import overload, register_jitable, intrinsic
from numba.core.unsafe.nrt import NRT_get_api
from numba.core.errors import TypingError
from enum import IntEnum
from numba.core import cgutils


class DTYPE_CODE(IntEnum):
    F64 = 0
    F32 = 1
    U64 = 2
    I64 = 3
    U32 = 4
    I32 = 5
    U16 = 6
    I16 = 7
    U8 = 8
    I8 = 9
    BOOL = 10


# Public lookup (NumPy dtype, Numba scalar type)
DTYPE_TABLE = {
    DTYPE_CODE.F64: (np.float64, types.float64),
    DTYPE_CODE.F32: (np.float32, types.float32),
    DTYPE_CODE.U64: (np.uint64, types.uint64),
    DTYPE_CODE.I64: (np.int64, types.int64),
    DTYPE_CODE.U32: (np.uint32, types.uint32),
    DTYPE_CODE.I32: (np.int32, types.int32),
    DTYPE_CODE.U16: (np.uint16, types.uint16),
    DTYPE_CODE.I16: (np.int16, types.int16),
    DTYPE_CODE.U8: (np.uint8, types.uint8),
    DTYPE_CODE.I8: (np.int8, types.int8),
    DTYPE_CODE.BOOL: (np.bool_, types.boolean),
}

# Map NumberClass -> (dtype_code:int, numpy_dtype_obj)
_NBSCALAR_TO_CODE = {
    types.float64: 0,
    types.float32: 1,
    types.uint64: 2,
    types.int64: 3,
    types.uint32: 4,
    types.int32: 5,
    types.uint16: 6,
    types.int16: 7,
    types.uint8: 8,
    types.int8: 9,
    types.boolean: 10,
}

_NUMPY_FROM_NBSCALAR = {
    types.float64: np.float64,
    types.float32: np.float32,
    types.uint64: np.uint64,
    types.int64: np.int64,
    types.uint32: np.uint32,
    types.int32: np.int32,
    types.uint16: np.uint16,
    types.int16: np.int16,
    types.uint8: np.uint8,
    types.int8: np.int8,
    types.boolean: np.bool_,
}


class ErrorCode(IntEnum):
    SUCCESS = 0
    PARSE_FAILED = -1
    INSUFFICIENT_BUFFERS = -2
    READ_INPUT_FAILED = -3
    FUNCTION_FAILED = -4
    OUTPUT_FORMAT_ERROR = -5
    WRONG_OUTPUT_COUNT = -6
    WRITE_OUTPUT_FAILED = -7


def get_error_message(error_code: int) -> str:
    messages = {
        ErrorCode.SUCCESS: "Success",
        ErrorCode.PARSE_FAILED: "Failed to parse function arguments",
        ErrorCode.INSUFFICIENT_BUFFERS: "Not enough buffers provided",
        ErrorCode.READ_INPUT_FAILED: "Failed to read input arrays (rank/dtype/NULL mismatch)",
        ErrorCode.FUNCTION_FAILED: "Function execution failed",
        ErrorCode.OUTPUT_FORMAT_ERROR: "Expected tuple output but got single value",
        ErrorCode.WRONG_OUTPUT_COUNT: "Wrong number of outputs returned",
        ErrorCode.WRITE_OUTPUT_FAILED: "Failed to write output arrays",
    }
    return messages.get(error_code, f"Unknown error code: {error_code}")


@njit
def get_nrt_api():
    return NRT_get_api()


@intrinsic
def _data_ptr(typingctx, arr_t):
    """Return the data pointer of a Numba array."""
    if not isinstance(arr_t, types.Array):
        return None
    sig = types.voidptr(arr_t)

    def codegen(context, builder, signature, args):
        ary = context.make_array(signature.args[0])(builder, args[0])
        return ary.data

    return sig, codegen


@intrinsic
def _meminfo_ptr(typingctx, arr_t):
    """The meminfo pointer of a Numba array.

    This pointer can be passed to Numba's NRT API functions for memory management.
    It is stored in the first entry of the Numba array structure.
    """
    if not isinstance(arr_t, types.Array):
        return None
    sig = types.voidptr(arr_t)

    def codegen(context, builder, signature, args):
        ary = context.make_array(signature.args[0])(context, builder, args[0])
        return ary.meminfo

    return sig, codegen


@intrinsic
def _uintp_to_voidptr(typingctx, v):
    """Cast a machine-sized unsigned integer (address) to a void* for carray."""
    if v != types.uintp:
        return None
    sig = types.voidptr(types.uintp)

    def codegen(context, builder, signature, args):
        uintp_val = args[0]
        voidptr_llty = context.get_value_type(types.voidptr)
        return builder.inttoptr(uintp_val, voidptr_llty)

    return sig, codegen


@intrinsic
def _voidptr_to_uintp(typingctx, v):
    if v != types.voidptr:
        return None
    sig = types.uintp(types.voidptr)

    def codegen(context, builder, signature, args):
        return builder.ptrtoint(args[0], context.get_value_type(types.uintp))

    return sig, codegen


def _numpy_dtype_from_code(code):
    if code == 0:
        return np.float64
    if code == 1:
        return np.float32
    if code == 2:
        return np.uint64
    if code == 3:
        return np.int64
    if code == 4:
        return np.uint32
    if code == 5:
        return np.int32
    if code == 6:
        return np.uint16
    if code == 7:
        return np.int16
    if code == 8:
        return np.uint8
    if code == 9:
        return np.int8
    if code == 10:
        return np.bool_
    return np.float64


def _dtype_code_from_numpy(dtype):
    if dtype == np.float64:
        return 0
    if dtype == np.float32:
        return 1
    if dtype == np.uint64:
        return 2
    if dtype == np.int64:
        return 3
    if dtype == np.uint32:
        return 4
    if dtype == np.int32:
        return 5
    if dtype == np.uint16:
        return 6
    if dtype == np.int16:
        return 7
    if dtype == np.uint8:
        return 8
    if dtype == np.int8:
        return 9
    if dtype == np.bool_:
        return 10
    raise TypeError("Unsupported dtype")


@njit(no_cpython_wrapper=True)
def read_bundle(
    size,
    buffers_ptr,
    meminfos_ptr,
    nbytes_ptr,
    ranks_ptr,
    shapes_ptr,
    strides_ptr,
    dtypes_ptr,
    modified_ptr,
):
    """Turn raw pointers into typed arrays and bundle them."""
    if size == 0:
        return (
            0,
            carray(buffers_ptr, (0,), dtype=np.uintp),
            carray(meminfos_ptr, (0,), dtype=np.uintp),
            carray(nbytes_ptr, (0,), dtype=np.uintp),
            carray(ranks_ptr, (0,), dtype=np.uintp),
            carray(shapes_ptr, (0,), dtype=np.uintp),
            carray(strides_ptr, (0,), dtype=np.uintp),
            carray(dtypes_ptr, (0,), dtype=np.uint8),
            carray(modified_ptr, (0,), dtype=np.uint8),
        )

    buffers = carray(buffers_ptr, (size,), dtype=np.uintp)
    meminfo = carray(meminfos_ptr, (size,), dtype=np.uintp)
    nbytes = carray(nbytes_ptr, (size,), dtype=np.uintp)
    ranks = carray(ranks_ptr, (size,), dtype=np.uintp)
    modified = carray(modified_ptr, (size,), dtype=np.uint8)

    total_axes = 0
    for i in range(size):
        total_axes += ranks[i]

    shapes = carray(shapes_ptr, (total_axes,), dtype=np.uintp)
    strides = carray(strides_ptr, (total_axes,), dtype=np.uintp)
    dtypes = carray(dtypes_ptr, (total_axes,), dtype=np.uint8)

    return (size, buffers, meminfo, nbytes, ranks, shapes, strides, dtypes, modified)


# ============================================================
# Public: wrappers
# ============================================================

# C signature for the VM entrypoints
c_sig = types.int64(
    types.uint64,  # size
    types.CPointer(types.c_uintp),  # buffers**
    types.CPointer(types.c_uintp),  # meminfos**
    types.CPointer(types.c_uintp),  # nbytes*
    types.CPointer(types.c_uintp),  # ranks*
    types.CPointer(types.c_uintp),  # shapes*
    types.CPointer(types.c_uintp),  # strides*
    types.CPointer(types.c_uint8),  # dtypes*
    types.CPointer(types.c_uint8),  # modified*
)


def wrap_njit_function(func, in_specs, out_specs):
    """
    Wrap an @njit function with typed/validated inputs and ensured outputs.

    in_specs/out_specs are tuples of (dtype_code:int, rank:int) — they must be
    literal tuples at the call sites inside the compiled wrapper.
    """
    in_specs = tuple((int(dc), int(r)) for (dc, r) in in_specs)
    out_specs = tuple((int(dc), int(r)) for (dc, r) in out_specs)
    n_inputs, n_outputs = len(in_specs), len(out_specs)

    @cfunc(c_sig)
    def entry(
        size,
        buffers_p,
        meminfos_p,
        nbytes_p,
        ranks_p,
        shapes_p,
        strides_p,
        dtypes_p,
        modified_p,
    ):
        # Parse bundle
        try:
            bundle = read_bundle(
                size,
                buffers_p,
                meminfos_p,
                nbytes_p,
                ranks_p,
                shapes_p,
                strides_p,
                dtypes_p,
                modified_p,
            )
        except Exception:
            return ErrorCode.PARSE_FAILED

        if size < n_inputs + n_outputs:
            return ErrorCode.INSUFFICIENT_BUFFERS

        # Inputs
        try:
            inputs = _build_inputs_checked(bundle, in_specs)
        except Exception:
            return ErrorCode.READ_INPUT_FAILED

        # Call user func
        try:
            outs = func(*inputs)
        except Exception:
            return ErrorCode.FUNCTION_FAILED

        # Normalize outputs
        if n_outputs == 1:
            if isinstance(outs, tuple):
                if len(outs) != 1:
                    return ErrorCode.WRONG_OUTPUT_COUNT
                outs = outs[0]
            outs = (outs,)
        else:
            if not isinstance(outs, tuple) or len(outs) != n_outputs:
                return ErrorCode.WRONG_OUTPUT_COUNT

        # Ensure & copy
        try:
            _ensure_and_copy_outputs(bundle, n_inputs, outs, out_specs)
        except Exception:
            return ErrorCode.WRITE_OUTPUT_FAILED

        return ErrorCode.SUCCESS

    return entry


def wrap_inplace_njit_function(func):
    """
    Wrap an in-place @njit func(bundle). Inputs in `in_specs` are validated.
    Any pointer/metadata writes must be done via ensure_array_from_bundle.
    """

    @cfunc(c_sig)
    def entry(
        size,
        buffers_p,
        meminfos_p,
        nbytes_p,
        ranks_p,
        shapes_p,
        strides_p,
        dtypes_p,
        modified_p,
    ):
        try:
            bundle = read_bundle(
                size,
                buffers_p,
                meminfos_p,
                nbytes_p,
                ranks_p,
                shapes_p,
                strides_p,
                dtypes_p,
                modified_p,
            )
        except Exception:
            return ErrorCode.PARSE_FAILED

        try:
            func(bundle)
        except Exception:
            return ErrorCode.FUNCTION_FAILED

        return ErrorCode.SUCCESS

    return entry


# Cast uintp -> void*
@intrinsic
def _uintp_to_voidptr(typingctx, v):
    if v != types.uintp:
        return None
    sig = types.voidptr(types.uintp)

    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], context.get_value_type(types.voidptr))

    return sig, codegen


# Data pointer (as uintp) from an array
@intrinsic
def _data_ptr_uintp(typingctx, arr_t):
    if not isinstance(arr_t, types.Array):
        return None
    sig = types.uintp(arr_t)

    def codegen(context, builder, signature, args):
        ary = context.make_array(signature.args[0])(builder, args[0])
        return builder.ptrtoint(ary.data, context.get_value_type(types.uintp))

    return sig, codegen


def view_from_bundle(bundle, index, dtype, rank):
    """Return a typed, validated STRIDED view of bundle.buffers[index]."""
    raise NotImplementedError


@overload(view_from_bundle, no_cpython_wrapper=True)
def _ov_view_from_bundle(bundle, index, dtype, rank):
    # rank must be a literal
    if not isinstance(rank, types.IntegerLiteral):
        raise TypingError("rank must be a literal int")
    rk = int(rank.literal_value)

    # dtype must be a NumPy scalar class (NumberClass)
    if isinstance(dtype, types.NumberClass):
        nb_scalar = dtype.instance_type
    else:
        raise TypingError("dtype must be a NumPy scalar class, e.g. np.float64")

    if nb_scalar not in _NBSCALAR_TO_CODE:
        raise TypingError("unsupported dtype")
    code = _NBSCALAR_TO_CODE[nb_scalar]
    np_dtype = _NUMPY_FROM_NBSCALAR[nb_scalar]

    base_shp = (0,) * rk
    base_std = (0,) * rk

    def impl(bundle, index, dtype, rank):
        size, buffers, meminfo, nbytes, ranks, shapes, strides, dtypes, modified = (
            bundle
        )
        if index >= size:
            print("index oob")
            raise ValueError("index OOB")
        if buffers[index] == 0:
            print("null buffer")
            raise ValueError("NULL buffer")
        if ranks[index] != rk:
            print("rank mismatch")
            raise ValueError("rank mismatch")
        if dtypes[index] != np.uint8(code):
            print("dtype mismatch")
            raise TypeError("dtype mismatch")

        base = 0
        for k in range(index):
            base += ranks[k]

        addr = buffers[index]

        if rk == 0:
            # scalar: just a 0-d view
            return carray(_uintp_to_voidptr(addr), (), np_dtype)

        # Build shape & strides tuples from bundle
        shp = base_shp
        std = base_std
        for ax in range(rk):
            shp = numba.np.unsafe.ndarray.tuple_setitem(shp, ax, shapes[base + ax])
            std = numba.np.unsafe.ndarray.tuple_setitem(std, ax, strides[base + ax])

        # Create a trivial base array and re-stride it
        base_arr = carray(_uintp_to_voidptr(addr), (1,), np_dtype)
        return np.lib.stride_tricks.as_strided(base_arr, shape=shp, strides=std)

    return impl


def ensure_array_from_bundle(bundle, index, shape, dtype, rank):
    """Ensure slot exists with (shape, dtype, rank). Set modified[index]=1 iff metadata written."""
    raise NotImplementedError


@overload(ensure_array_from_bundle, no_cpython_wrapper=True)
def _ov_ensure_array_from_bundle(bundle, index, shape, dtype, rank):
    # rank must be literal
    if not isinstance(rank, types.IntegerLiteral):
        raise TypingError("rank must be a literal int")
    rk = int(rank.literal_value)

    # dtype must be a NumberClass
    if isinstance(dtype, types.NumberClass):
        nb_scalar = dtype.instance_type
    else:
        raise TypingError("dtype must be a NumPy scalar class, e.g. np.float64")

    if nb_scalar not in _NBSCALAR_TO_CODE:
        raise TypingError("unsupported dtype")
    code = _NBSCALAR_TO_CODE[nb_scalar]
    np_dtype = _NUMPY_FROM_NBSCALAR[nb_scalar]

    # shape must be a fixed-length UniTuple(intp, rk)
    if not (
        isinstance(shape, types.UniTuple)
        and shape.dtype == types.intp
        and shape.count == rk
    ):
        raise TypingError("shape must be a UniTuple(intp, rank)")

    zero = (0,) * rk

    def impl(bundle, index, shape, dtype, rank):
        size, buffers, meminfo, nbytes, ranks, shapes, strides, dtypes, modified = (
            bundle
        )
        if index >= size:
            raise ValueError("index OOB")

        base = 0
        for k in range(index):
            base += ranks[k]

        # Check existing
        ok = True
        if buffers[index] == 0 or meminfo[index] == 0:
            ok = False
        if ranks[index] != rk:
            ok = False
        if dtypes[index] != np.uint8(code):
            ok = False
        for ax in range(rk):
            if shapes[base + ax] != shape[ax]:
                ok = False

        addr = buffers[index]

        if ok:
            if rk == 0:
                return carray(_uintp_to_voidptr(addr), (), np_dtype)
            # Build strides tuple and return strided view
            std = zero
            for ax in range(rk):
                std = numba.np.unsafe.ndarray.tuple_setitem(std, ax, strides[base + ax])
            base_arr = carray(_uintp_to_voidptr(addr), (1,), np_dtype)
            return np.lib.stride_tricks.as_strided(base_arr, shape=shape, strides=std)

        arr = np.zeros(shape, dtype=np_dtype)

        # write pointers/metadata (store real NRT meminfo for Rust)
        # buffers[index]  = _data_ptr(arr)
        buffers[index] = arr.ctypes.data
        meminfo[index] = _voidptr_to_uintp(_meminfo_ptr(arr))
        ranks[index] = rk
        dtypes[index] = np.uint8(code)
        nbytes[index] = np.uintp(arr.nbytes)
        for ax in range(rk):
            shapes[base + ax] = arr.shape[ax]
            strides[base + ax] = arr.strides[ax]
        modified[index] = np.uint8(1)

        addr = buffers[index]
        if rk == 0:
            return carray(_uintp_to_voidptr(addr), (), np_dtype)

        # return a strided view using the bundle's freshly-written strides
        std = zero
        for ax in range(rk):
            std = numba.np.unsafe.ndarray.tuple_setitem(std, ax, strides[base + ax])
        base_arr = carray(_uintp_to_voidptr(addr), (1,), np_dtype)
        return np.lib.stride_tricks.as_strided(base_arr, shape=shape, strides=std)

    return impl
