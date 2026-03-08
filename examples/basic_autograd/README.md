# basic_autograd

core/autograd の基本的な使い方を示すデモ。Variable を使って簡単な線形計算 `loss = sum(x * w + b)` を行い、backward で各変数の勾配を自動計算する。

## Run

```bash
moon run examples/basic_autograd
```

## Expected output

```
=== Forward ===
x:    Tensor(shape=[3], data=[1, 2, 3])
w:    Tensor(shape=[3], data=[0.5, -1, 2])
b:    Tensor(shape=[3], data=[0.1, 0.2, 0.3])
loss: Tensor(shape=[], data=[5.1])

=== Backward (gradients) ===
x.grad: Tensor(shape=[3], data=[0.5, -1, 2])
w.grad: Tensor(shape=[3], data=[1, 2, 3])
b.grad: Tensor(shape=[3], data=[1, 1, 1])
```
