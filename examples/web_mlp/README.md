# web_mlp

Train a 2-layer MLP with SGD directly in the browser via WebAssembly (wasm-gc).
The MoonBit source is identical to `basic_mlp` — no code changes needed to run in the browser.

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

The page loads the WASM module and runs 100 training steps.
Training progress is displayed both on the page and in the browser console:

```
step 0: loss = Tensor(shape=[], data=[137.27...])
step 10: loss = Tensor(shape=[], data=[111.71...])
step 20: loss = Tensor(shape=[], data=[69.84...])
...
step 90: loss = Tensor(shape=[], data=[0.45...])
Training complete.
```

Loss decreases from ~137 to ~0.45 over 100 steps — the model learns `y = 2*x1 + 3*x2` from 3 data points, entirely inside the browser.

## How it works

1. `moon build --target wasm-gc` compiles MoonBit to a WebAssembly module
2. `index.html` loads the `.wasm` file and calls the exported `_start` function
3. MoonBit's `println` outputs characters via `spectest.print_char`, which the JS glue buffers and renders on the page
