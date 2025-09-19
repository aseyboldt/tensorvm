#!/usr/bin/env python3
"""
PyTensor RT VM Usage Examples

This file demonstrates how to use the PyTensor RT VM for high-performance
numerical computations with seamless Python/Numba integration.

Note: This example assumes the Rust library has been compiled and is available.
Some parts may not run until the full integration is complete.
"""

import numba
import numpy as np
import pytensor_rt
from pytensor_rt import pytensor_rt as lib

nrt = pytensor_rt.numba_ffi.get_nrt_api()


@numba.jit(no_cpython_wrapper=True, nopython=True, fastmath=True)
def func(bundle):
    x = pytensor_rt.numba_ffi.view_from_bundle(bundle, 0, np.float64, 2)
    y = pytensor_rt.numba_ffi.view_from_bundle(bundle, 1, np.float64, 2)
    #y = pytensor_rt.numba_ffi.ensure_array_from_bundle(bundle, 1, x.shape, np.float64, 2)

    #assert x.strides[-1] == 8
    #assert y.strides[-1] == 8
    #x = np.ascontiguousarray(x)
    #y = np.ascontiguousarray(y)
    #y[...] = (2 * x) * (2 * x)

    n, m = x.shape
    assert x.shape == y.shape
    for i in range(n):
        for j in range(m):
            z = x[i, j]
            y[i, j] = (2 * z) * (2 * z)

    return 0

mul = pytensor_rt.numba_ffi.wrap_inplace_njit_function(func)
builder = lib.VMBuilder()

z = np.random.randn(10, 10)

x = lib.Variable.new_array(z, "C")
y = lib.Variable.new_array(z, "C")
z = lib.Variable.new_array(z, "C")

builder.add_external_call(mul.address, [x, y], func)
builder.add_external_call(mul.address, [y, z], func)

vm = builder.build(nrt)
vm.run()
out = z.values()
