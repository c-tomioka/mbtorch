# web_mlp

Compare optimizer and activation function combinations for a 2-layer MLP,
running entirely in the browser via WebAssembly (wasm-gc).

## What it does

Trains a Linear(2,4) → activation → Linear(4,1) model on synthetic data
(y = 2\*x1 + 3\*x2) using 6 combinations:

| Optimizer | Activation |
|-----------|------------|
| SGD (lr=0.001) | ReLU, tanh, sigmoid |
| Adam (lr=0.05) | ReLU, tanh, sigmoid |

Each combination trains for 100 steps with identical initial weights
(same random seeds for fair comparison). The page displays a summary table
of final loss values alongside the full training log.

## Why this demo

The initial version of this demo ran a single SGD + ReLU training loop to
prove that **MoonBit + WASM can run ML in the browser**. This expanded
version goes further: it compares 6 optimizer × activation combinations
side-by-side so you can **see how these choices affect convergence** — all
computed inside the browser with no server-side processing.

## Prerequisites

- [MoonBit toolchain](https://www.moonbitlang.com/) installed (`moon` command available)
- A wasm-gc capable browser:
  - Chrome 119+
  - Firefox 120+
  - Safari 18.2+

## Build

From the project root:

```bash
moon build --target wasm-gc
```

This generates `_build/wasm-gc/debug/build/examples/web_mlp/web_mlp.wasm`.

## Run

Start a local HTTP server from the **project root** (not from this directory):

```bash
python3 -m http.server 8080
```

Then open: http://localhost:8080/examples/web_mlp/index.html

## What you will see

The page runs all 6 combinations sequentially. For each one, training
progress is shown both on the page and in the browser console:

```
=== SGD + ReLU ===
step 0: loss = Tensor(shape=[], data=[137.27...])
step 10: loss = Tensor(shape=[], data=[111.71...])
...
Training complete.
=== SGD + tanh ===
...
=== Adam + sigmoid ===
...
Training complete.
```

A summary table at the top shows the final loss for each combination:

| Combination | Final Loss |
|-------------|-----------|
| SGD + ReLU | ~0.45 |
| SGD + tanh | ~4.39 |
| SGD + sigmoid | ~14.71 |
| Adam + ReLU | ~0.53 |
| Adam + tanh | ~4.56 |
| Adam + sigmoid | ~4.52 |

### Reading the results

- **ReLU converges fastest** for this task — its gradient passes through unchanged, so the optimizer can make large updates efficiently.
- **sigmoid struggles** because its output is clamped to (0, 1), while the targets are 5, 7, and 8. The final Linear layer must learn very large weights to compensate, slowing convergence.
- **Adam rescues sigmoid**: SGD + sigmoid barely moves (14.71), but Adam + sigmoid reaches 4.52 — Adam's adaptive learning rate helps overcome the vanishing gradients.
- **Adam + ReLU slightly worse than SGD + ReLU** (0.53 vs 0.45): Adam's momentum can cause oscillation on this small, simple problem. More data or more steps would likely close the gap.

These patterns illustrate why activation and optimizer choice matters — and why there is no single "best" combination for all tasks.

## How it works

1. `moon build --target wasm-gc` compiles MoonBit to a WebAssembly module
2. `index.html` loads the `.wasm` file and calls the exported `_start` function
3. MoonBit's `println` outputs characters via `spectest.print_char`, which the JS glue buffers and renders on the page
4. The JS parses each output line to extract labels and loss values, then builds a results summary table
