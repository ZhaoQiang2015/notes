# TORCH

torch package包含多维张量的数据结构，以及在其上的数据操作，并提供了很多有用的多类型的方法。

## Tensors

### `torch.is_tensor(obj)`

        return True if obj is a PyTorch tensor.
* `Parameters`
  * **obj**(Object) - Object to test

### `torch.is_storage(obj)`

        returns True if obj is a PyTorch storage object.
* `Parameters`
  * **obj**(Object) - Object to test

### `torch.is_floating_point(tensor) -> (bool)`

        returns True if the data type of tensor is floating i.e., `torch.float64`, `torch.float32`, `torch.float16`.
* `Parameters`
  * **tensor**(Tensor) - the PyTorch tensor to test

### `torch.set_default_dtype(d)`
        
        set the default floating point dtype to d. used as default floating point type for type inference in `torch.tensor()`.
* `Parameters`
  * **d**(torch.dtype) - the floating point dtype to mark the default

### `torch.get_default_dtype() -> torch.dtype`

        get the current default floating point torch.dtype

```python
>>> torch.tensor([1.2, 3]).dtype    # initial default for floating point is torch.float32
torch.float32
>>> torch.set_default_dtype(torch.float64)
>>> torch.tensor([1.2, 3]).dtype    # a new floating point tensor
torch.float64
>>> torch.get_default_dtype()
torch.float64

```

### `torch.set_default_tensor_type(t)`

        set the default `torch.Tensor` type to floating point tensor type t. The default floating point tensor type is initially `torch.FloatTensor`.
* `Parameters`
  * **t****(type or string) - the floating point tensor type or its name

### `torch.numel(input) -> int`

        returns the total number of elements in the input tensor. 张量总元素个数
* `Parameters`
  * **input**(Tensor) - the input tensor

### `torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)`

        打印选项，参考自numpy
* `Parameters`
  * **precision** - 小数位精度(default = 4)
  * **threshold** - total number of array elements which trigger summarization rather than full repr(default = 1000)
  * **edgeitems** - number of array items in summary at beginning and end of each dimension(default = 3)
  * **linewidth** - number of characters per line for the purpose of inserting line breaks(default = 80)
  * **profile** - sane defaults for pretty printing
  * **sci_mode** - Enable(True) or disable(False) scientific notation.

### `torch.set_flush_denormal(mode) -> bool`

        disables denormal floating numbers on CPU. returns True if your system supports flushing denormal numbers and it successfully configures flush denormal mode. it only supported on x86 architectures supporting SSE3.
* `Parameters`
  * **mode**(bool) - control whether to enable flush denormal mode or not

## Creation Ops

### `torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor`

        constructs a tensor with data. 此方法会深拷贝，
* `Parameters`
  * **data**(array_like) - initial data for the tensor. 可以为list, tuple, NumPy ndarray, scalar, and other types.
  * **dtype**(`torch.dtype`, optional) - desired data type of returned tensor. default: if `None`, infers data type from `data`
  * **device**(`torch.device`, optional) - desired device of returned tensor.
  * **requires_grad**(bool, optional) - if autograd should record operations on the returned tensor. default: `False`
  * **pin_memory**(bool, optional) - if set, returned tensor would be allocated in the pinned memory. works for CPU tensors, default: `False`

### `torch.sparse_coo_tensor(indices, values, size=None, dtype=None, device=None, requires_grad=False) -> Tensor`

        construct a sparse tensors in COO(rdinate) format with non-zero elements at the given `indices` with the given `values`.
* `Parameters`
  * **indices**(array_like) - initial data for the tensor. can be list, tuple, NumPy ndarray, scalar and other types
  * **values**(array_like) - initial values for the tensor. can be list, tuple, NumPy ndarray, scalar and other types
  * **size**(list, tuple, or `torch.Size`, optional) - size of sparse tensor.
  * **dtype**(`torch.dtype`, optional) - desired data type of returned tensor. default: if None, infers data type from `values`
  * **device**(`torch.device`, optional) - desired device of returned tensor
  * **requires_grad**(bool, optional) - if autograd should record operations on the returned tensor, default: `False`

### `torch.as_tensor(data, dtype=None, device=None) -> Tensor`

        转换data -> `torch.Tensor`, 如果data为相同类型，相同device的tensor, 不会产生拷贝。否则，一个新的tensor会产生。若为在CPU上同类型的`ndarray`，也不会产生拷贝。
* `Parameters`
  * **data**(array_like) - initial data for the tensor
  * **dtype**(`torch.dtype`, optional) - desired data type of returned tensor.
  * **device**(`torch.device`, optional)

### `torch.from_numpy(ndarray) -> Tensor`

        create a tensor from a `numpy.ndarray`, 返回的tensor与ndarray共享内存。返回的tensor不能resizable

```python
>>> a = numpy.array([1,2,3])
>>> t = torch.from_numpy(a)
>>> t
tensor([1,2,3])
>>> t[0] = -1
>>> a
array([-1,2,3])
```

### `torch.zeros(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`

        return a tensor filled with the scalar value 0, with the shape defined by the variable `sizes`
* `Parameters`
  * **sizes**(int...) - a sequence of integers defining the shape of the output tensor.
  * **out**(Tensor, optional) - the output tensor
  * **dtype**(`torch.dtype`, optional) - desired data type
  * **layout**(`torch.layout`, optional) - desired layout of return Tensor
  * **device**(`torch.device`, optional) - desired device of return tensor
  * **requires_grad**(bool, optional) - default: `False`

```python
>>> torch.zeros(2, 3)
torsor([[0., 0., 0.],
        [0., 0., 0.]])
>>> torch.zeros(3)
tensor([0., 0., 0.])
```

