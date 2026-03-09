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
(same random seeds for fair comparison). The page displays:

1. An **interactive loss chart** with 6 color-coded curves (log scale)
2. **Checkboxes** to show/hide individual curves
3. A **results summary table** with final loss values
4. A collapsible **training log** with step-by-step output

## Prerequisites

- [MoonBit toolchain](https://www.moonbitlang.com/) installed (`moon` command available)
- A wasm-gc capable browser:
  - Chrome 119+
  - Firefox 120+
  - Safari 18.2+

## Build & Run

From the project root:

```bash
moon build --target wasm-gc
python3 -m http.server 8080
```

Open: http://localhost:8080/examples/web_mlp/index.html

## What you'll see

1. Open the page and click **"Train 6 models in your browser"**.
2. The browser trains 6 MLP variants (SGD/Adam × ReLU/tanh/sigmoid) from scratch via WebAssembly — no server needed.
3. Once training finishes, a **loss chart** with 6 color-coded curves appears.
4. Use **checkboxes** to show/hide individual curves.
5. A **results table** summarizes the final loss for each combination.
6. Click **Training Log** to view the full step-by-step output.
7. Click **"Train again"** to re-run all 6 training runs from scratch.

### Example results

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
2. `index.html` pre-compiles the `.wasm` module on page load; clicking the train button instantiates it and calls `_start`
3. MoonBit's `println` outputs characters via `spectest.print_char`, which the JS glue buffers and renders
4. The MoonBit code emits `DATA:label,step,loss` tagged lines on every step for chart data
5. JS parses these lines to build Chart.js datasets, and renders the interactive loss chart with checkbox controls
6. Human-readable `step N: loss = ...` lines are shown in the collapsible training log
