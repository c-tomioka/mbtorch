# basic_mlp

2-layer MLP training demo comparing SGD and Adam optimizers.

## What it does

- Builds a small MLP: `Linear(2, 4) → ReLU → Linear(4, 1)`
- Trains on synthetic data (`y = 2*x1 + 3*x2`) for 100 steps
- Runs training twice with the same initial weights:
  1. **SGD** (lr=0.001) — classic gradient descent
  2. **Adam** (lr=0.05) — adaptive learning rate
- Prints loss every 10 steps so you can compare convergence

## Run

```bash
moon run examples/basic_mlp
```

## Expected output

```
=== SGD (lr=0.001) ===
step 0: loss = Tensor(shape=[], data=[137.27...])
step 10: loss = Tensor(shape=[], data=[111.71...])
step 20: loss = Tensor(shape=[], data=[69.84...])
...
step 90: loss = Tensor(shape=[], data=[0.45...])
=== Adam (lr=0.05) ===
step 0: loss = Tensor(shape=[], data=[137.27...])
step 10: loss = Tensor(shape=[], data=[74.44...])
step 20: loss = Tensor(shape=[], data=[4.81...])
...
step 90: loss = Tensor(shape=[], data=[0.53...])
Training complete.
```

Both optimizers reduce loss. SGD converges smoothly, while Adam drops faster in early steps but may oscillate slightly before settling.
