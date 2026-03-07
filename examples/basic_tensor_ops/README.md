# basic_tensor_ops

Demonstrates the core Tensor API: constructors (`scalar`, `from_array`, `from_array_2d`, `zeros`, `ones`), element-wise operations (`+`, `*`), and matrix multiplication (`matmul`).

## Run

```bash
moon run examples/basic_tensor_ops
```

## Expected output

```
=== Constructors ===
scalar:        Tensor(shape=[], data=[42])
from_array:    Tensor(shape=[3], data=[1, 2, 3])
from_array_2d: Tensor(shape=[2, 2], data=[1, 2, 3, 4])
zeros([2,3]):  Tensor(shape=[2, 3], data=[0, 0, 0, 0, 0, 0])
ones([3]):     Tensor(shape=[3], data=[1, 1, 1])

=== Element-wise ops ===
a:     Tensor(shape=[3], data=[1, 2, 3])
b:     Tensor(shape=[3], data=[4, 5, 6])
a + b: Tensor(shape=[3], data=[5, 7, 9])
a * b: Tensor(shape=[3], data=[4, 10, 18])

=== Matmul ===
x:            Tensor(shape=[2, 2], data=[1, 2, 3, 4])
y:            Tensor(shape=[2, 2], data=[5, 6, 7, 8])
x.matmul(y):  Tensor(shape=[2, 2], data=[19, 22, 43, 50])
```
