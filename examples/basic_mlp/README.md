# basic_mlp

2-layer MLP training demo using MbTorch.

## What it does

- Builds a small MLP: `Linear(2, 4) → ReLU → Linear(4, 1)`
- Trains on synthetic data (`y = 2*x1 + 3*x2`) for 100 steps
- Uses hand-written SGD (lr=0.001) to update parameters
- Prints loss every 10 steps — you should see loss decreasing over time

## Run

```bash
moon run examples/basic_mlp
```

## Expected output

```
step 0: loss = ...
step 10: loss = ...
step 20: loss = ...
...
step 90: loss = ...
Training complete.
```

Loss values will decrease as training progresses.
