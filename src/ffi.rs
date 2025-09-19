use std::{ffi::c_void, sync::Arc};

use crate::array::{DType, NumbaBuffer};

/// Opaque struct representing Numba's MemInfo
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct NRT_MemInfo {
    _private: [u8; 0], // opaque
}
#[allow(non_camel_case_types)]
type NRT_managed_dtor = unsafe extern "C" fn(*mut c_void);
#[allow(non_camel_case_types)]
type NRT_allocate = unsafe extern "C" fn(nbytes: usize) -> *mut NRT_MemInfo;
#[allow(non_camel_case_types)]
type NRT_manage_memory =
    unsafe extern "C" fn(data: *mut c_void, dtor: NRT_managed_dtor) -> *mut NRT_MemInfo;
#[allow(non_camel_case_types)]
type NRT_acquire = unsafe extern "C" fn(mi: *mut NRT_MemInfo);
#[allow(non_camel_case_types)]
type NRT_release = unsafe extern "C" fn(mi: *mut NRT_MemInfo);
#[allow(non_camel_case_types)]
type NRT_get_data = unsafe extern "C" fn(mi: *mut NRT_MemInfo) -> *mut c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NumbaRuntime {
    pub(crate) allocate: NRT_allocate,
    pub(crate) manage_memory: NRT_manage_memory,
    pub(crate) acquire: NRT_acquire,
    pub(crate) release: NRT_release,
    pub(crate) get_data: NRT_get_data,
}

#[repr(C)]
pub struct CallCommandArg<'buffer> {
    // The total number of buffers
    size: usize,
    // Pointer to an array of pointers to the buffers
    // The length of this array is `size`
    // The pointers can be null if the buffer is not allocated
    buffers: *mut *mut c_void,
    // numba nrt MemInfo for each of the buffers. Might be null.
    meminfo: *mut *mut NRT_MemInfo,
    // Pointer to an array of nbytes of the buffers
    nbytes: *mut usize,
    // Pointer to an array of ranks of the buffers
    ranks: *const usize,
    // The shapes of the buffers. This contains sum(ranks) elements
    shapes: *mut usize,
    // The strides of the buffers. This contains sum(ranks) elements
    strides: *mut usize,
    // The dtype codes of the buffers. This contains sum(ranks) elements
    dtypes: *mut u8,
    // An output flag for each buffer, indicating if the metadata (shape,
    // strides, data, meminfo) was modified by the function.
    modified: *mut u8,
    _phantom: std::marker::PhantomData<&'buffer ()>,
}

pub struct CArgBuffer {
    buffers: Vec<*mut c_void>,
    meminfo: Vec<*mut NRT_MemInfo>,
    // Pointer to an array of nbytes of the buffers
    nbytes: Vec<usize>,
    // Pointer to an array of ranks of the buffers
    ranks: Vec<usize>,
    // The shapes of the buffers. This contains sum(ranks) elements
    shapes: Vec<usize>,
    // The strides of the buffers. This contains sum(ranks) elements
    strides: Vec<usize>,
    // The dtype codes of the buffers. This contains sum(ranks) elements
    modified: Vec<u8>,
    dtypes: Vec<u8>,
}

impl CArgBuffer {
    pub fn clear(&mut self) {
        let Self {
            buffers,
            meminfo,
            nbytes,
            ranks,
            shapes,
            strides,
            modified,
            dtypes,
        } = self;
        buffers.clear();
        meminfo.clear();
        nbytes.clear();
        ranks.clear();
        shapes.clear();
        strides.clear();
        modified.clear();
        dtypes.clear();
    }

    pub fn push_empty(&mut self, rank: usize, dtype: DType) {
        self.buffers.push(std::ptr::null_mut());
        self.meminfo.push(std::ptr::null_mut());
        self.nbytes.push(0);
        self.ranks.push(rank);
        self.dtypes.push(dtype.type_code());
        self.modified.push(0);
        for _ in 0..rank {
            self.shapes.push(0);
            self.strides.push(0);
        }
    }

    pub fn push_buffer(
        &mut self,
        ptr: *mut c_void,
        meminfo: *mut NRT_MemInfo,
        nbytes: usize,
        shape: &[usize],
        strides: &[usize],
        dtype: DType,
    ) {
        let rank = shape.len();
        assert!(strides.len() == rank);
        self.buffers.push(ptr);
        self.meminfo.push(meminfo);
        self.nbytes.push(nbytes);
        self.ranks.push(rank);
        self.dtypes.push(dtype.type_code());
        self.modified.push(0);
        self.shapes.extend_from_slice(shape);
        self.strides.extend_from_slice(strides);
    }

    pub(crate) fn push_buffer_from_info(&mut self, info: BufferInfo) {
        self.push_buffer(
            info.ptr,
            info.meminfo,
            info.nbytes,
            info.shape,
            info.strides,
            info.dtype,
        );
    }

    pub(crate) fn buffers<'a>(&'a self) -> impl Iterator<Item = BufferInfo<'a>> {
        let mut shape_idx = 0;
        self.buffers.iter().enumerate().map(move |(i, &ptr)| {
            let rank = self.ranks[i];
            let shape = &self.shapes[shape_idx..shape_idx + rank];
            let stride = &self.strides[shape_idx..shape_idx + rank];
            shape_idx += rank;
            BufferInfo {
                ptr,
                meminfo: self.meminfo[i],
                nbytes: self.nbytes[i],
                shape,
                strides: stride,
                dtype: DType::from_code(self.dtypes[i]),
                modified: self.modified[i],
            }
        })
    }

    pub(crate) fn new() -> Self {
        Self {
            buffers: Vec::new(),
            meminfo: Vec::new(),
            nbytes: Vec::new(),
            ranks: Vec::new(),
            shapes: Vec::new(),
            strides: Vec::new(),
            modified: Vec::new(),
            dtypes: Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BufferInfo<'a> {
    pub(crate) ptr: *mut c_void,
    pub(crate) meminfo: *mut NRT_MemInfo,
    pub(crate) nbytes: usize,
    pub(crate) shape: &'a [usize],
    pub(crate) strides: &'a [usize],
    pub(crate) dtype: DType,
    pub(crate) modified: u8,
}

impl BufferInfo<'_> {
    pub fn is_modified(&self) -> bool {
        self.modified != 0
    }

    pub fn to_buffer(&self, numba_runtime: Arc<NumbaRuntime>) -> NumbaBuffer {
        NumbaBuffer::from_buffer_info(self, numba_runtime)
    }
}

/// Call a function pointer with the given inputs.
///
/// The function must uphold the following contract:
///
/// - It must have the signature `extern "C" fn(CPtrArg) -> i64`
/// - It must return 0 on success, and non-zero on failure.
/// - It must return the number of modified buffers in `num_modified`
/// - All modified buffers must be at the end of the list of buffers.
/// - It can replace the buffer with new ones, but it must
///   then change the corresponding meminfo pointer. It must be
///   a valid numba nrt MemInfo pointer.
/// - The returned buffers together with shape and stride have
///   to be valid.
/// - It must not free any modified buffers. The caller
///   will take care of that.
/// - If it returns an error code, it must not modify any buffers.
#[derive(Clone, Debug)]
pub struct ExternalArrayFunc {
    func_ptr: CallCommandFn,
}

impl ExternalArrayFunc {
    pub fn new(func_ptr: CallCommandFn) -> Self {
        Self { func_ptr }
    }

    /// Call the function with the given arguments.
    ///
    /// On success, returns the number of modified buffers.
    /// On failure, returns an error code.
    pub fn call(&self, args: &mut CArgBuffer) -> Result<usize, i64> {
        let ret = (self.func_ptr)(
            args.buffers.len(),
            args.buffers.as_mut_ptr(),
            args.meminfo.as_mut_ptr(),
            args.nbytes.as_mut_ptr(),
            args.ranks.as_ptr(),
            args.shapes.as_mut_ptr(),
            args.strides.as_mut_ptr(),
            args.dtypes.as_mut_ptr(),
            args.modified.as_mut_ptr(),
        );
        if ret < 0 {
            Err(ret)
        } else {
            Ok(ret as usize)
        }
    }
}

type CallCommandFn = extern "C" fn(
    size: usize,
    buffers: *mut *mut c_void,
    meminfos: *mut *mut NRT_MemInfo,
    nbytes: *mut usize,
    ranks: *const usize,
    shapes: *mut usize,
    stides: *mut usize,
    dtypes: *mut u8,
    modified: *mut u8,
) -> i64;
