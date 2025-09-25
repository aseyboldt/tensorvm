use std::{ffi::c_void, fmt::Display, sync::Arc};

use aligned_vec::{avec, avec_rt, AVec, ConstAlign};
use anyhow::{Context, Result};
use ndarray::{ArrayViewD, ShapeBuilder};
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArray};
use pyo3::{Py, Python};
use smallvec::SmallVec;

use crate::ffi::{BufferInfo, NRT_MemInfo, NumbaRuntime};

const INLINE_SIZE: usize = 4;

// TODO verify what type numba uses for shape and strides
pub type Shape = SmallVec<[usize; INLINE_SIZE]>;
pub type Strides = SmallVec<[usize; INLINE_SIZE]>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Order {
    RowMajor,
    ColumnMajor,
    Arbitrary,
}

impl Default for Order {
    fn default() -> Self {
        Self::RowMajor
    }
}

impl Order {
    pub fn c() -> Self {
        Self::RowMajor
    }

    pub fn f() -> Self {
        Self::ColumnMajor
    }
}

impl TryFrom<&str> for Order {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        match value {
            "C" | "c" => Ok(Self::RowMajor),
            "F" | "f" => Ok(Self::ColumnMajor),
            "A" | "a" => Ok(Self::Arbitrary),
            _ => anyhow::bail!("Unknown order string: {}", value),
        }
    }
}

impl Display for Order {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Order::RowMajor => "C",
            Order::ColumnMajor => "F",
            Order::Arbitrary => "A",
        };
        write!(f, "{}", s)
    }
}

impl From<ndarray::Order> for Order {
    fn from(value: ndarray::Order) -> Self {
        match value {
            ndarray::Order::RowMajor => Order::RowMajor,
            ndarray::Order::ColumnMajor => Order::ColumnMajor,
            _ => Order::Arbitrary,
        }
    }
}

impl TryInto<ndarray::Order> for Order {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<ndarray::Order, Self::Error> {
        match self {
            Order::RowMajor => Ok(ndarray::Order::RowMajor),
            Order::ColumnMajor => Ok(ndarray::Order::ColumnMajor),
            Order::Arbitrary => anyhow::bail!("Cannot convert Arbitrary order to ndarray::Order"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub dtype: DType,
    pub order: Order,
    pub shape: Option<SmallVec<[Option<usize>; INLINE_SIZE]>>,
}

impl TensorType {
    pub fn rank(&self) -> Option<usize> {
        match &self.shape {
            Some(shape) => Some(shape.len()),
            None => None,
        }
    }
}

impl Display for TensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape_str = match &self.shape {
            None => "[...]".to_string(),
            Some(shape) => {
                let shape: Vec<String> = shape
                    .iter()
                    .map(|len| match len {
                        Some(d) => d.to_string(),
                        None => "?".to_string(),
                    })
                    .collect();
                format!("[{}]", shape.join(", "))
            }
        };
        write!(
            f,
            "TensorType(dtype={}, shape={}, order={})",
            self.dtype, shape_str, self.order
        )
    }
}

#[derive(Debug)]
pub struct NumbaBuffer {
    ptr: *mut c_void,
    // This contains the numba nrt MemInfo pointer,
    // which keeps a refcount for the buffer stored
    // in self.ptr.
    meminfo: *mut NRT_MemInfo,
    nbytes: usize,
    byte_strides: Strides,
    numba_runtime: Arc<NumbaRuntime>,
}

// SAFETY: NumbaBuffer can be sent between threads
// because NRT_MemInfo is thread-safe.
unsafe impl Send for NumbaBuffer {}
unsafe impl Sync for NumbaBuffer {}

impl NumbaBuffer {
    pub(crate) fn from_buffer_info(
        arg: &BufferInfo<'_>,
        runtime: Arc<NumbaRuntime>,
    ) -> NumbaBuffer {
        Self {
            ptr: arg.ptr,
            meminfo: arg.meminfo,
            nbytes: arg.nbytes,
            byte_strides: Strides::from_slice(arg.strides),

            numba_runtime: runtime,
        }
    }

    fn element_strides(&self, dtype: DType) -> impl Iterator<Item = usize> + '_ {
        // TODO checks
        self.byte_strides
            .iter()
            .map(move |s| s / dtype.size_bytes())
    }
}

impl Drop for NumbaBuffer {
    fn drop(&mut self) {
        unsafe { (self.numba_runtime.release)(self.meminfo) };
    }
}

#[derive(Debug)]
pub struct NativeBuffer {
    data: AVec<u8, ConstAlign<256>>,
    byte_strides: Strides,
    offset: usize,
}

impl NativeBuffer {
    fn element_strides(&self, dtype: DType) -> impl Iterator<Item = usize> + '_ {
        // TODO checks
        let size = dtype.size_bytes();
        self.byte_strides.iter().map(move |s| s / size)
    }

    pub(crate) fn new(dtype: DType, shape: &[usize], order: Order) -> Self {
        let item_count = shape.iter().fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .expect("Shape dimensions overflow when calculating item count")
        });
        let capacity = item_count
            .checked_mul(dtype.size_bytes())
            .expect("nbytes overflow");
        // DLpack uses 256-byte alignment for its arrays
        let data = avec![[256] | 0u8; capacity];
        Self {
            data,
            byte_strides: calculate_strides_for_new_array(shape, dtype.size_bytes(), order),
            offset: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
    U64,
    I64,
    U32,
    I32,
    U16,
    I16,
    U8,
    I8,
    Bool,
}

impl Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DType::F64 => "float64",
            DType::F32 => "float32",
            DType::U64 => "uint64",
            DType::I64 => "int64",
            DType::U32 => "uint32",
            DType::I32 => "int32",
            DType::U16 => "uint16",
            DType::I16 => "int16",
            DType::U8 => "uint8",
            DType::I8 => "int8",
            DType::Bool => "bool",
        };
        write!(f, "{}", s)
    }
}

pub trait PrimitiveAsDType {
    fn dtype() -> DType;
}

macro_rules! impl_primitive_as_dtype {
    ($($typ:ty => $dtype:expr),* $(,)?) => {
        $(
            impl PrimitiveAsDType for $typ {
                fn dtype() -> DType {
                    $dtype
                }
            }
        )*
    };
}

// We don't implement for bool, because not all bit patterns are valid bools.
impl_primitive_as_dtype! {
    f64 => DType::F64,
    f32 => DType::F32,
    u64 => DType::U64,
    i64 => DType::I64,
    u32 => DType::U32,
    i32 => DType::I32,
    u16 => DType::U16,
    i16 => DType::I16,
    u8 => DType::U8,
    i8 => DType::I8,
}

impl DType {
    pub fn type_code(&self) -> u8 {
        match self {
            DType::F64 => 0,
            DType::F32 => 1,
            DType::U64 => 2,
            DType::I64 => 3,
            DType::U32 => 4,
            DType::I32 => 5,
            DType::U16 => 6,
            DType::I16 => 7,
            DType::U8 => 8,
            DType::I8 => 9,
            DType::Bool => 10,
        }
    }

    pub fn from_code(code: u8) -> Self {
        match code {
            0 => DType::F64,
            1 => DType::F32,
            2 => DType::U64,
            3 => DType::I64,
            4 => DType::U32,
            5 => DType::I32,
            6 => DType::U16,
            7 => DType::I16,
            8 => DType::U8,
            9 => DType::I8,
            10 => DType::Bool,
            _ => panic!("Unknown dtype code: {}", code),
        }
    }

    pub fn from_string(s: &str) -> Option<Self> {
        match s {
            "float64" | "f64" => Some(DType::F64),
            "float32" | "f32" => Some(DType::F32),
            "int64" | "i64" => Some(DType::I64),
            "uint64" | "u64" => Some(DType::U64),
            "int32" | "i32" => Some(DType::I32),
            "uint32" | "u32" => Some(DType::U32),
            "int16" | "i16" => Some(DType::I16),
            "uint16" | "u16" => Some(DType::U16),
            "int8" | "i8" => Some(DType::I8),
            "uint8" | "u8" => Some(DType::U8),
            "bool" => Some(DType::Bool),
            _ => None,
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F64 => 8,
            DType::F32 => 4,
            DType::U64 => 8,
            DType::I64 => 8,
            DType::U32 => 4,
            DType::I32 => 4,
            DType::U16 => 2,
            DType::I16 => 2,
            DType::U8 => 1,
            DType::I8 => 1,
            DType::Bool => 1,
        }
    }
}

#[derive(Debug)]
pub enum Buffer {
    Empty,
    Numba(NumbaBuffer),
    Native(NativeBuffer),
}

impl Buffer {
    /// # Safety
    /// Must be called with the correct type T that matches the dtype of the buffer.
    unsafe fn as_typed_slice_mut<T: PrimitiveAsDType>(&mut self) -> Result<Option<&mut [T]>> {
        match self {
            Buffer::Numba(ref buf) => {
                let nelements = buf.nbytes.checked_div(std::mem::size_of::<T>()).unwrap();
                Ok(Some(unsafe {
                    std::slice::from_raw_parts_mut(buf.ptr as *mut T, nelements)
                }))
            }
            Buffer::Native(ref mut buf) => Ok(Some(unsafe {
                let nelements = buf
                    .data
                    .len()
                    .checked_div(std::mem::size_of::<T>())
                    .unwrap();
                std::slice::from_raw_parts_mut(
                    buf.data.as_mut_ptr().add(buf.offset) as *mut T,
                    nelements,
                )
            })),
            Buffer::Empty => Ok(None),
        }
    }
    /// # Safety
    /// Must be called with the correct type T that matches the dtype of the buffer.
    unsafe fn as_typed_slice<T: PrimitiveAsDType>(&self) -> Result<Option<&[T]>> {
        match self {
            Buffer::Numba(ref buf) => {
                let nelements = buf.nbytes.checked_div(std::mem::size_of::<T>()).unwrap();
                Ok(Some(unsafe {
                    std::slice::from_raw_parts(buf.ptr as *const T, nelements)
                }))
            }
            Buffer::Native(ref buf) => Ok(Some(unsafe {
                let nelements = buf
                    .data
                    .len()
                    .checked_div(std::mem::size_of::<T>())
                    .unwrap();
                std::slice::from_raw_parts(buf.data.as_ptr().add(buf.offset) as *const T, nelements)
            })),
            Buffer::Empty => Ok(None),
        }
    }
}

#[derive(Debug)]
pub struct Tensor {
    pub(crate) buffer: Buffer,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
}

/// Conversion to ndarray views
impl Tensor {
    pub fn as_array_view<'a, T: PrimitiveAsDType>(
        &'a self,
    ) -> Result<Option<ndarray::ArrayView<'a, T, ndarray::IxDyn>>> {
        if T::dtype() != self.dtype {
            anyhow::bail!(
                "DType mismatch: expected {:?}, got {:?}",
                T::dtype(),
                self.dtype
            );
        }

        // SAFETY: We trust that the NumbaBuffer was created correctly
        // SAFETY: We checked that the dtype matches T
        let buffer = unsafe { self.buffer.as_typed_slice::<T>()? };

        let Some(buffer) = buffer else {
            return Ok(None);
        };

        let mut strides = SmallVec::<[usize; INLINE_SIZE]>::with_capacity(self.shape.len());
        match &self.buffer {
            Buffer::Numba(buf) => strides.extend(buf.element_strides(self.dtype)),
            Buffer::Native(buf) => strides.extend(buf.element_strides(self.dtype)),
            Buffer::Empty => {
                unreachable!()
            }
        };

        let strides = ndarray::IxDyn(strides.as_slice());
        let shape_with_strides = ndarray::IxDyn(self.shape.as_slice()).strides(strides);

        // SAFETY: We trust that the NumbaBuffer was created correctly
        Ok(Some(ndarray::ArrayView::from_shape(
            shape_with_strides,
            buffer,
        )?))
    }

    pub fn as_array_view_mut<'a, T: PrimitiveAsDType>(
        &'a mut self,
    ) -> Result<Option<ndarray::ArrayViewMut<'a, T, ndarray::IxDyn>>> {
        if T::dtype() != self.dtype {
            anyhow::bail!(
                "DType mismatch: expected {:?}, got {:?}",
                T::dtype(),
                self.dtype
            );
        }

        let mut strides = SmallVec::<[usize; INLINE_SIZE]>::with_capacity(self.shape.len());
        match &self.buffer {
            Buffer::Numba(buf) => strides.extend(buf.element_strides(self.dtype)),
            Buffer::Native(buf) => strides.extend(buf.element_strides(self.dtype)),
            Buffer::Empty => {
                unreachable!()
            }
        };
        let strides = ndarray::IxDyn(strides.as_slice());

        // SAFETY: We trust that the NumbaBuffer was created correctly
        // SAFETY: We checked that the dtype matches T
        let buffer = unsafe { self.buffer.as_typed_slice_mut::<T>() }
            .context("Could not convert variable buffer to slice")?;
        let Some(buffer) = buffer else {
            return Ok(None);
        };

        let shape_with_strides = ndarray::IxDyn(self.shape.as_slice()).strides(strides);

        // SAFETY: We trust that the NumbaBuffer was created correctly
        Ok(Some(
            ndarray::ArrayViewMut::from_shape(shape_with_strides, buffer)
                .context("Failed to create array view")?,
        ))
    }

    pub(crate) fn buffer_info(&self) -> BufferInfo {
        match &self.buffer {
            Buffer::Numba(buf) => BufferInfo {
                ptr: buf.ptr,
                strides: buf.byte_strides.as_slice(),
                nbytes: buf.nbytes,
                meminfo: buf.meminfo,
                shape: self.shape.as_slice(),
                dtype: self.dtype,
                modified: 0,
            },
            Buffer::Native(buf) => BufferInfo {
                ptr: buf.data.as_ptr().wrapping_add(buf.offset) as *mut c_void,
                strides: buf.byte_strides.as_slice(),
                nbytes: buf.data.len(),
                meminfo: std::ptr::null_mut(),
                shape: self.shape.as_slice(),
                dtype: self.dtype,
                modified: 0,
            },
            Buffer::Empty => BufferInfo {
                ptr: std::ptr::null_mut(),
                strides: self.shape.as_slice(),
                nbytes: 0,
                meminfo: std::ptr::null_mut(),
                shape: self.shape.as_slice(),
                dtype: self.dtype,
                modified: 0,
            },
        }
    }

    pub fn new_empty(rank: usize, inner: DType) -> Self {
        Self {
            buffer: Buffer::Empty,
            shape: SmallVec::from_elem(0, rank),
            dtype: inner,
        }
    }

    pub fn to_numpy(&self) -> Result<Option<Py<PyUntypedArray>>> {
        macro_rules! to_numpy_impl {
            ($typ:ty, $self:expr) => {
                Python::attach(|py| {
                    let Some(view) = $self.as_array_view::<$typ>()? else {
                        return Ok(None);
                    };
                    let array = PyArrayDyn::from_owned_array(py, view.to_owned());
                    let array = array.as_untyped();
                    let array = array.clone().unbind();
                    Ok(Some(array))
                })
            };
        }

        match self.dtype {
            DType::F64 => to_numpy_impl!(f64, self),
            DType::F32 => to_numpy_impl!(f32, self),
            DType::U64 => to_numpy_impl!(u64, self),
            DType::I64 => to_numpy_impl!(i64, self),
            DType::U32 => to_numpy_impl!(u32, self),
            DType::I32 => to_numpy_impl!(i32, self),
            DType::U16 => to_numpy_impl!(u16, self),
            DType::I16 => to_numpy_impl!(i16, self),
            DType::U8 => to_numpy_impl!(u8, self),
            DType::I8 => to_numpy_impl!(i8, self),
            // TODO: Support bool
            DType::Bool => {
                panic!("Bool not supported for to_numpy")
            }
        }
    }

    pub fn from_numpy<'py, T: PrimitiveAsDType + numpy::Element + Clone>(
        array: PyReadonlyArrayDyn<'py, T>,
        order: Order,
    ) -> Result<Self> {
        let array = array.as_array();
        Self::from_ndarray(array, order)
    }

    pub fn from_ndarray<T: PrimitiveAsDType + numpy::Element + Clone>(
        values: ArrayViewD<'_, T>,
        order: Order,
    ) -> Result<Self> {
        // Create a new buffer and copy data over
        let mut array = Tensor::new(T::dtype(), values.shape(), order);
        let mut array_view = array
            .as_array_view_mut::<T>()
            .context("Failed to create array view of variable")?
            .unwrap();
        array_view.assign(&values);
        Ok(array)
    }

    pub fn new(dtype: DType, shape: &[usize], order: Order) -> Self {
        let buffer = NativeBuffer::new(dtype, shape, order);
        Self {
            buffer: Buffer::Native(buffer),
            shape: SmallVec::from_slice(shape),
            dtype,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn order(&self) -> Option<Order> {
        // TODO clean this up and make sure there are no overflows
        match &self.buffer {
            Buffer::Numba(buf) => {
                if buf.byte_strides.is_empty() || self.shape.is_empty() {
                    return Some(Order::Arbitrary);
                }
                let ndim = self.shape.len();
                if buf.byte_strides[ndim - 1] == buf.nbytes / self.shape[ndim - 1] {
                    Some(Order::RowMajor)
                } else if buf.byte_strides[0] == buf.nbytes / self.shape[0] {
                    Some(Order::ColumnMajor)
                } else {
                    Some(Order::Arbitrary)
                }
            }
            Buffer::Native(buf) => {
                if buf.byte_strides.is_empty() || self.shape.is_empty() {
                    return Some(Order::Arbitrary);
                }
                let ndim = self.shape.len();
                if buf.byte_strides[ndim - 1] == self.dtype.size_bytes() {
                    Some(Order::RowMajor)
                } else if buf.byte_strides[0] == self.dtype.size_bytes() {
                    Some(Order::ColumnMajor)
                } else {
                    Some(Order::Arbitrary)
                }
            }
            Buffer::Empty => Some(Order::Arbitrary),
        }
    }

    pub(crate) fn as_scalar_bool(&self) -> Result<bool> {
        // Check if this is a scalar (0-dimensional tensor)
        if !self.shape.is_empty() {
            anyhow::bail!("Tensor is not a scalar, shape: {:?}", self.shape);
        }

        // Handle different dtypes that can be interpreted as boolean
        match self.dtype {
            DType::Bool => {
                todo!("bool arrays are not implemented fully");
            }
            DType::I32 => {
                let array_view = self.as_array_view::<i32>()?;
                match array_view {
                    Some(view) => {
                        if view.is_empty() {
                            anyhow::bail!("Scalar tensor has no data");
                        }
                        Ok(view[ndarray::IxDyn(&[])] != 0)
                    }
                    None => anyhow::bail!("Tensor buffer is empty"),
                }
            }
            DType::I64 => {
                let array_view = self.as_array_view::<i64>()?;
                match array_view {
                    Some(view) => {
                        if view.is_empty() {
                            anyhow::bail!("Scalar tensor has no data");
                        }
                        Ok(view[ndarray::IxDyn(&[])] != 0)
                    }
                    None => anyhow::bail!("Tensor buffer is empty"),
                }
            }
            DType::U32 => {
                let array_view = self.as_array_view::<u32>()?;
                match array_view {
                    Some(view) => {
                        if view.is_empty() {
                            anyhow::bail!("Scalar tensor has no data");
                        }
                        Ok(view[ndarray::IxDyn(&[])] != 0)
                    }
                    None => anyhow::bail!("Tensor buffer is empty"),
                }
            }
            DType::U64 => {
                let array_view = self.as_array_view::<u64>()?;
                match array_view {
                    Some(view) => {
                        if view.is_empty() {
                            anyhow::bail!("Scalar tensor has no data");
                        }
                        Ok(view[ndarray::IxDyn(&[])] != 0)
                    }
                    None => anyhow::bail!("Tensor buffer is empty"),
                }
            }
            _ => anyhow::bail!(
                "Tensor dtype cannot be interpreted as boolean, got: {:?}",
                self.dtype
            ),
        }
    }

    pub(crate) fn as_shape(&self) -> Result<Shape> {
        // Check if dtype is appropriate for shape (should be integer)
        match self.dtype {
            DType::I64 | DType::I32 | DType::U64 | DType::U32 => {}
            _ => anyhow::bail!(
                "Tensor dtype is not suitable for shape, got: {:?}",
                self.dtype
            ),
        }

        // Shape tensor should be 1-dimensional
        if self.shape.len() != 1 {
            anyhow::bail!(
                "Shape tensor must be 1-dimensional, got shape: {:?}",
                self.shape
            );
        }

        let shape_len = self.shape[0];
        let mut result_shape = Shape::with_capacity(shape_len);

        // Extract shape values based on dtype
        match self.dtype {
            DType::I64 => {
                let array_view = self.as_array_view::<i64>()?;
                if let Some(view) = array_view {
                    for &dim in view.iter() {
                        if dim < 0 {
                            anyhow::bail!("Negative dimension in shape: {}", dim);
                        }
                        result_shape.push(dim as usize);
                    }
                } else {
                    anyhow::bail!("Tensor buffer is empty");
                }
            }
            DType::I32 => {
                let array_view = self.as_array_view::<i32>()?;
                if let Some(view) = array_view {
                    for &dim in view.iter() {
                        if dim < 0 {
                            anyhow::bail!("Negative dimension in shape: {}", dim);
                        }
                        result_shape.push(dim as usize);
                    }
                } else {
                    anyhow::bail!("Tensor buffer is empty");
                }
            }
            DType::U64 => {
                let array_view = self.as_array_view::<u64>()?;
                if let Some(view) = array_view {
                    for &dim in view.iter() {
                        result_shape.push(dim as usize);
                    }
                } else {
                    anyhow::bail!("Tensor buffer is empty");
                }
            }
            DType::U32 => {
                let array_view = self.as_array_view::<u32>()?;
                if let Some(view) = array_view {
                    for &dim in view.iter() {
                        result_shape.push(dim as usize);
                    }
                } else {
                    anyhow::bail!("Tensor buffer is empty");
                }
            }
            _ => unreachable!("Already checked dtype above"),
        }

        Ok(result_shape)
    }
}

// Helper function to calculate byte strides for a new array with given shape and order
fn calculate_strides_for_new_array(shape: &[usize], element_size: usize, order: Order) -> Strides {
    let ndim = shape.len();
    let mut strides: Strides = SmallVec::with_capacity(ndim);

    if ndim == 0 {
        return strides;
    }

    // Initialize with correct length
    strides.resize(ndim, 0);

    match order {
        // Row-major (C-order): last axis is contiguous; base stride is element_size in bytes
        Order::RowMajor | Order::Arbitrary => {
            strides[ndim - 1] = element_size;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1]
                    .checked_mul(shape[i + 1])
                    .expect("Overflow in stride calculation");
            }
        }
        // Column-major (F-order): first axis is contiguous; base stride is element_size in bytes
        Order::ColumnMajor => {
            strides[0] = element_size;
            for i in 1..ndim {
                strides[i] = strides[i - 1]
                    .checked_mul(shape[i - 1])
                    .expect("Overflow in stride calculation");
            }
        }
    }

    strides
}
