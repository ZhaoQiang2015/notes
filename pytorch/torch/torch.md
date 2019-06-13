# TORCH

torch package包含多维张量的数据结构，以及在其上的数据操作，并提供了很多有用的多类型的方法。

## Tensors

> `torch.is_tensor(obj)`

return True if obj is a PyTorch tensor.

* `Parameters`
  * **obj**(Object) - Object to test

> `torch.is_storage(obj)`

returns True if obj is a PyTorch storage object.

* `Parameters`
  * **obj**(Object) - Object to test

> `torch.is_floating_point(tensor) -> (bool)`

returns True if the data type of tensor is floating i.e., `torch.float64`, `torch.float32`, `torch.float16`.

* `Parameters`
  * **tensor**(Tensor) - the PyTorch tensor to test

> `torch.set_default_dtype(d)`

set the default floating point dtype to d. used as default floating point type for type inference in `torch.tensor()`.

* `Parameters`
  * **d**(torch.dtype) - the floating point dtype to mark the default

> `torch.get_default_dtype() -> torch.dtype`

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

> `torch.set_default_tensor_type(t)`

set the default `torch.Tensor` type to floating point tensor type t. The default floating point tensor type is initially `torch.FloatTensor`.

* `Parameters`
  * **t****(type or string) - the floating point tensor type or its name

> `torch.numel(input) -> int`

returns the total number of elements in the input tensor. 张量总元素个数

* `Parameters`
  * **input**(Tensor) - the input tensor

> `torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)`

打印选项，参考自numpy

* `Parameters`
  * **precision** - 小数位精度(default = 4)
  * **threshold** - total number of array elements which trigger summarization rather than full repr(default = 1000)
  * **edgeitems** - number of array items in summary at beginning and end of each dimension(default = 3)
  * **linewidth** - number of characters per line for the purpose of inserting line breaks(default = 80)
  * **profile** - sane defaults for pretty printing
  * **sci_mode** - Enable(True) or disable(False) scientific notation.

> `torch.set_flush_denormal(mode) -> bool`

disables denormal floating numbers on CPU. returns True if your system supports flushing denormal numbers and it successfully configures flush denormal mode. it only supported on x86 architectures supporting SSE3.

* `Parameters`
  * **mode**(bool) - control whether to enable flush denormal mode or not

## Creation Ops

> `torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor`

constructs a tensor with data. 此方法会深拷贝，

* `Parameters`
  * **data**(array_like) - initial data for the tensor. 可以为list, tuple, NumPy ndarray, scalar, and other types.
  * **dtype**(`torch.dtype`, optional) - desired data type of returned tensor. default: if `None`, infers data type from `data`
  * **device**(`torch.device`, optional) - desired device of returned tensor.
  * **requires_grad**(bool, optional) - if autograd should record operations on the returned tensor. default: `False`
  * **pin_memory**(bool, optional) - if set, returned tensor would be allocated in the pinned memory. works for CPU tensors, default: `False`

> `torch.sparse_coo_tensor(indices, values, size=None, dtype=None, device=None, requires_grad=False) -> Tensor`

construct a sparse tensors in COO(rdinate) format with non-zero elements at the given `indices` with the given `values`.

* `Parameters`
  * **indices**(array_like) - initial data for the tensor. can be list, tuple, NumPy ndarray, scalar and other types
  * **values**(array_like) - initial values for the tensor. can be list, tuple, NumPy ndarray, scalar and other types
  * **size**(list, tuple, or `torch.Size`, optional) - size of sparse tensor.
  * **dtype**(`torch.dtype`, optional) - desired data type of returned tensor. default: if None, infers data type from `values`
  * **device**(`torch.device`, optional) - desired device of returned tensor
  * **requires_grad**(bool, optional) - if autograd should record operations on the returned tensor, default: `False`

> `torch.as_tensor(data, dtype=None, device=None) -> Tensor`

转换data -> `torch.Tensor`, 如果data为相同类型，相同device的tensor, 不会产生拷贝。否则，一个新的tensor会产生。若为在CPU上同类型的`ndarray`，也不会产生拷贝。

* `Parameters`
  * **data**(array_like) - initial data for the tensor
  * **dtype**(`torch.dtype`, optional) - desired data type of returned tensor.
  * **device**(`torch.device`, optional)

> `torch.from_numpy(ndarray) -> Tensor`

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

> `torch.zeros(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`

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

> `torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor`

return a tensor filled with the scalar value 0, with the same size as `input`

* `Parameters`
  * **input**(Tensor) - the size of `input` will be determine size of the output tensor
  * **dtype**(`torch.dtype`, optional) - desired data type of returned Tensor. default: if `None`, defaults to the dtype of `input`
  * **layout**(`torch.layout`, optional) - desired layout of the returned Tensor
  * **device**(`torch.device`, optional) - desired device of returned Tensor
  * **requires_grad**(bool, optional) - if autograd should record operations on the returned tensor. default: `False`

> `torch.ones(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`

return a tensor filled with the scalar value 1.

* `Parameters`
  * **sizes**(int...) - a sequence of integers defining the shape of the output tensor. can be variable number of arguments or a collection like a list or tuple
  * **out**(Tensor, optional) - output tensor
  * **dtype**(`torch.dtype`, optional) - desired data type of returned tensor
  * **layout**(`torch.layout`, optional) - desired layout of returned tensor. default: `torch.strided`
  * **device**(`torch.device`, optional) - desired device of returned tensor
  * **requires_grad**(bool, optional) - if autograd should record operations on the returned tensor. default: `False`

> `torch.ones_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor`

return a tensor filled with the scalar value 1, with the same size as `input`. `torch.ones_like(input)` is equivalent to `torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)`.
* `Parameters`
  * **input**(Tensor) - the size of `input` will determine size of the output tensor
  * **dtype**(`torch.dtype`, optional) - the desired data type of returned tensor
  * **layout**(`torch.layout`, optional) - desired layout of returned tensor
  * **device**(`torch.device`, optional) - desired device of returned tensor
  * **requires_grad**(bool, optional)

> `torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`

return a 1-D tensor of size floor((end-start)/step) with values from the interval [start, end) taken with common difference step begining from start.
* `Parameters`
  * **start**(Number) - start value for the set of points. default: 0
  * **end**(Number) - ending value for the set of points
  * **step**(Number) - the gap between each pair of adjacent points. default: 1
  * **out**(Tensor, optional) - output tensor
  * **dtype**(`torch.dtype`, optional) - the desired data type of returned tensor
  * **layout**(`torch.layout, optional) - desired layout
  * **device**(`torch.device, optional) - desired device
  * **requires_grad**(bool, optional) - default: `False`

> `torch.range(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`

return a 1-D tensor of size floor((end-start)/step)+1.

* `Parameters`

  * **start**(float) - start value, default: 0
  * **end**(float) - ending value
  * **step**(float) - default: 1
  * **out**(Tensor, optional) - output tensor
  * **dtype**(`torch.dtype`, optional) - 
  * **layout**(`torch.layout`, optional) -
  * **device**(`torch.device`, optional) -
  * **requires_grad**(bool, optional) -

> `torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`

return a 1-D tensor of steps equally spaced points between start and end, size if `steps`.

* `Parameters`
  * **start**(float) - starting value
  * **end**(float) - ending value
  * **steps**(int) - number of points to sample between `start` and `end`, default: 100
  * **out**(Tensor, optional) - output tensor
  * **dtype**(`torch.dtype`, optional) - 
  * **layout**(`torch.layout`, optional) -
  * **device**(`torch.device`, optional) -
  * **requires_grad**(bool, optional) -

> `torch.logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`

return a 1-D tensor of `steps` size points logarithmically spaced with base `base` between base^start to base^end.

* `Parameters`
  * **start**(float)
  * **end**(float)
  * **steps**(int) - number of points to sample. default: 100
  * **base**(float) - base of the logarithm function. default: 10.0
  * **out**(Tensor, optional)
  * **dtype**(`torch.dtype`, optional)
  * **layout**(`torch.layout`, optional)
  * **device**(`torch.device`, optional)
  * **requires_grad**(bool, optional)

```python
>>> torch.logspace(start=-10, end=10, steps=5)
tensor([1.0000e-10, 1.0000e-05, 1.0000e+00, 1.0000e+05, 1.0000e+10])
>>> torch.logspace(start=2, end=2, steps=1, base=2)
tensor([4.0])
```

> `torch.eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`

return a 2-D tensor with ones on the diagonal and zeros elsewhere.

* `Parameters`
  * **n**(int) - 行数
  * **m**(int, optional) - 列数, default: =n
  * **out**(Tensor, optional)
  * **dtype**(`torch.dtype`, optional)
  * **layout**(`torch.layout`, optional)
  * **device**(`torch.device`, optional)
  * **requires_grad**(bool, optional)

> `torch.empty(*sizes, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor`

return a tensor filled with uninitialized data.

* `Parameters`
  * **sizes**(int...) - 
  * **out**(Tensor, optional)
  * **dtype**(`torch.dtype, optional)
  * **layout**(`torch.layout`, optional)
  * **device**(`torch.device`, optional)
  * **requires_grad**(bool, optional)
  * **pin_memory**(bool, optional) - if set, returned tensor would be allocated in the pinned memory. works only for CPU tensors. default: `False`

> `torch.empty_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor`

return a uninitialized tensor with the same size as `input`.

* **input**(Tensor)
* **dtype**(`torch.dtype`, optional)
* **layout**(`torch.layout`, optional)
* **device**(`torch.device`, optional)
* **requires_grad**(bool, optional)

> `torch.full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`

return a tensor of size `size` filled with `fill_value`.

* `Parameters`
  * **size**(int...) - a list, tuple or `torch.Size` of integers defining the shape of the output tensor
  * **fill_value** - the number to fill the output tensor with
  * **out**(Tensor, optional)
  * **dtype**(`torch.dtype`, optional)
  * **layout**(`torch.layout`, optional)
  * **device**(`torch.device`, optional)
  * **requires_grad**(`bool, optional)

> `torch.full_like(input, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor`

和上面的类似

* `Parameters`
  * **input**(Tensor)
  * **fill_value** - 要填充的值
  * **dtype**(`torch.dtype`, optional)
  * **layout**(`torch.layout`, optional)
  * **device**(`torch.device`, optional)
  * **requires_grad**(bool, optional)

## Indexing, Slicing, Joining, Mutating Ops

> `torch.cat(tensors, dim=0, out=None) -> Tensor`

在指定维度上连接tensors, 所有tensors必须有相同的shape, 可以视为`torch.split()`的逆操作。

* `Parameters`
  * **tensors**(sequence of Tensors) - 相同类型的tensors，
  * **dim**(int, optional) - 指定的要连接的维度
  * **out**(Tensor, optional)

> `torch.chunk(tensor, chunks, dim=0) -> Tensor`

split a tensor into a specific number  of chunks.

* `Parameters`
  * **tensor**(Tensor) - 要split的tensor
  * **chunks**(int) - 返回的chunks个数
  * **dim**(int) - split的维度

> `torch.gather(input, dim, index, out=None, sparse_grad=False) -> Tensor`

* `Parameters`
  * **input**(Tensor) - source tensor
  * **dim**(int) - the axis along which to index
  * **index**(LongTensor) - the indices of elements to gather
  * **out**(Tensor, optional)
  * **sparse_grad**(bool, optional) - if `True`, `input` will be a sparse tensor.

```python
>>> t = torch.tensor([[1, 2], [3, 4]])
>>> torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
tensor([1, 1],
        [4, 3])
```

> `torch.index_select(input, dim, index, out=None) -> Tensor`

从指定维度按索引取`input`中的数据

* `Parameters`
  * **input**(Tensor)
  * **dim**(int)
  * **index**(LongTensor)
  * **out**(Tensor, optional)

> `torch.masked_select(input, mask, out=None) -> Tensor`

根据`mask`返回一个1-D tensor，`mask`与`input`的shape可以不一致, 不共享内存，新拷贝一份。

* `Parameters`
  * **input**(Tensor)
  * **mask**(ByteTensor) - the tensor containing the binary mask to index with 
  * **out**(Tensor, optional)

> `torch.narrow(input, dimension, start, length) -> Tensor`

从指定维度`dimension`上在`start`与`start+length`之间取数据，共享内存。

* `Parameters`
  * **input**(Tensor)
  * **dimension**(int)
  * **start**(int)
  * **length**(int)

> `torch.nonzero(input, out=None) -> Tensor`

返回非零元素的下标索引

* `Parameters`
  * **input**(Tensor)
  * **out**(LongTensor, optional)

> `torch.reshape(input, shape) -> Tensor`

* `Parameters`
  * **input**(Tensor)
  * **shape**(tuple of python:ints) - new shape

> `torch.split(tensor, split_size_or_sections, dim=0)`

沿指定维度`dim`对`tensor`分块，若`split_size_or_sections`是整数，整数分块，若是`list`，按`list`分块

* `Parameters`
  * **tensor**(Tensor)
  * **split_size_or_sections**(int or list(int)) - size of a single chunk or list of sizes for each_chunk
  * **dim**(int)

> `torch.squeeze(input, dim=None, out=None) -> Tensor`

return a tensor with all dimensions of `input` of size 1 removed.
if input is of shape(A$\times$1$\times$B$\times$C$\times$1$\times$D) then the out tensor will be of shape: (A$\times$B$\times$C$\times$D).
若`dim`给定，则在指定维度上操作，
**共享内存**

* `Parameters`
  * **input**(Tensor)
  * **dim**(int, optional)
  * **out**(Tensor, optional)

> `torch.stack(seq, dim=0, out=None) -> Tensor`

沿着指定维度拼接tensors

* `Parameters`
  * **seq**(sequence of Tensors)
  * **dim**(int)
  * **out**(Tensor, optional)

> `torch.t(input) -> Tensor`

0-D和1-D tensor转置不变，2-D转置等价于`transpose(input, 0, 1)`

* `Parameters`
  * **input**(Tensor)

> `torch.take(input, indices) -> Tensor`

从指定`indices`上取`input`中的元素，`input`被当成1-D向量看待。

* `Parameters`
  * **input**(Tensor)
  * **indices**(LongTensor)

```python
>>> src = torch.tensor([[4,3,5],
                        [6,7,8]])
>>> torch.take(src, torch.tensor([0,2,5]))
tensor([4,5,8])
```

> `torch.transpose(input, dim0, dim1) -> Tensor`

交换指定维度，共享内存

* `Parameters`
  * **input**(Tensor)
  * **dim0**(int)
  * **dim1**(int)

> `torch.unbind(tensor, dim=0) -> seq`

删除指定维度，返回一个`tuple`的切片在给定的维度上。

* `Parameters`
  * **tensor**(Tensor)
  * **dim**(int)

> `torch.unsqueeze(input, dim, out=None) -> Tensor`

在指定位置rxty维度大小为1，共享内存。
`dim`的取值范围`[-input.dim()-1, input.dim()+1)`

* `Parameters`
  * **input**(Tensor)
  * **dim**(int)
  * **out**(Tensor, optional)

> `torch.where(condition, x, y) -> Tensor`

return a tensor of elements selected from either `x` or `y`, depending on `condition`
if condition:   $x_i$
else:           $y_i$

* `Parameters`
  * **condition**(ByteTensor) - when true(nonzero), yield x, otherwise yield y
  * **x**(Tensor)
  * **y**(Tensor)
* return a tensor of shape equal to the broadcasted shape of `condition`

## Random sampling

> `torch.manual_seed(seed)`