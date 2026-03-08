# web_mlp

Train a 2-layer MLP with SGD directly in the browser via WebAssembly (wasm-gc). Same training code as `basic_mlp`, demonstrating that MbTorch runs in the browser with zero source-level changes.

## Prerequisites

- Chrome 119+ / Firefox 120+ / Safari 18.2+ (wasm-gc support)

## Build

```bash
moon build --target wasm-gc
```

## Run

Start a local HTTP server from the project root:

```bash
python3 -m http.server 8080
```

Then open: http://localhost:8080/examples/web_mlp/index.html

## Expected output

The page displays training progress (loss logged every 10 steps) and finishes with "Training complete.":

```
step 0: loss = ...
step 10: loss = ...
...
step 90: loss = ...
Training complete.
```

Loss decreases over the 100 training steps, same as the CLI version.
