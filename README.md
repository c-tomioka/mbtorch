# MbTorch

Lightweight AI/ML framework in MoonBit. Train, fine-tune, and run neural networks directly in the browser and on edge devices via WebAssembly. Privacy-first, serverless ML for everyone.

## Overview

**MbTorch** is an open-source machine learning framework built in MoonBit, designed for lightweight models and on-device adaptation. It focuses on enabling users to build small-to-medium neural networks from scratch in MoonBit, import existing pretrained models (via ONNX and safetensors), and run training and inference efficiently on browsers and edge devices through WebAssembly.

Unlike traditional ML frameworks that assume powerful servers and large-scale training, MbTorch embraces constrained environments and offline usage. The goal is to let users obtain an optimized model for their own data and device, entirely on-device and without sending private data to the cloud.

### Why MbTorch?

Most AI frameworks prioritize massive, server-side models. MbTorch takes a different approach:

- **Train on-device**: Keep sensitive data local; no mandatory cloud backend.
- **Lightweight by design**: Target practical models suitable for CPU/WASM, rather than giant distributed training setups.
- **WebAssembly-first**: Compile MoonBit to WASM and run anywhere a WebAssembly runtime is available (browsers, edge runtimes, serverless).
- **Privacy-preserving**: Support personal fine-tuning and adaptation without uploading raw data.  
- **Interoperable**: Import models via ONNX and safetensors to reuse existing ecosystems and tooling.
- **MoonBit native**: Leverage MoonBit’s AI-friendly, WASM-oriented design, small binaries, and fast compilation for edge AI workflows.

### Target Use Cases

MbTorch focuses on small and mid-sized neural networks and edge-centric tasks, for example:

- Handwritten digit or character recognition in the browser  
- Sensor data anomaly detection on IoT devices  
- Simple recommendation or ranking systems on-device  
- Local personalization and LoRA-style fine-tuning of imported models  
- Educational ML projects and interactive browser demos  

## Project Status

MbTorch has reached its **first MVP**: tensors, reverse-mode autograd, `Linear`/MLP layers, SGD optimizer, and JSON-based model serialization are all implemented and tested (78 tests passing). You can define, train, save, and restore a small neural network entirely in MoonBit today.

**Next up:** ONNX import (subset), safetensors weight loading, binary `.mbt` format, Adam optimizer, and additional layer types.

## Features

### Implemented ✅

- **Tensor Operations** — Multi-dimensional tensors with add, sub, mul, scale, matmul, transpose, sum; constructors for scalar, 1D, 2D, zeros, ones, rand
- **Automatic Differentiation** — Reverse-mode autograd engine (`Variable`, `backward`) with gradient support for add/sub/mul/matmul/sum/relu/expand_rows; includes `grad_check` for numerical verification
- **Neural Network Layers** — `Linear` layer (weight, bias, forward, `from_tensors` for deserialization); composable into multi-layer MLPs with ReLU activation
- **Optimizers** — `SGD` optimizer with `new`, `step()`, `zero_grad()`
- **Model I/O** — Serialize/deserialize models to `.mbt`-style JSON via `serialize_model` / `deserialize_model`; roundtrip-tested with trained models
- **Examples** — 4 working demos: `basic_tensor_ops`, `basic_autograd`, `basic_mlp` (training), `save_mlp` (serialization roundtrip)

### Planned

- **ONNX Import** — Graph structure and operator import for small MLPs/CNNs
- **safetensors Import** — Efficient, safe weight loading from HuggingFace ecosystem
- **Binary `.mbt` Format** — Compact binary serialization (metadata + packed tensor buffer)
- **Additional Optimizers** — Adam, learning rate schedulers
- **Additional Layers** — Conv2d, attention, batch normalization
- **Data Utilities** — Mini-batching helpers and preprocessing
- **Edge & Browser Runtime** — WASM-targeted inference and lightweight fine-tuning

## Quick Start

> MbTorch is in early development. APIs are experimental and subject to change.

```moonbit
// Train a 2-layer MLP with SGD (this code runs today!)
fn main {
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
  )
  let target = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[5.0], [7.0], [8.0]]),
  )

  let params = l1.parameters()
  params.append(l2.parameters())
  let optimizer = @optim.SGD::new(params, lr=0.001)

  for _step = 0; _step < 100; _step = _step + 1 {
    let pred = l2.forward(l1.forward(x).relu())
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()  // MSE
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  }
}
```

More examples:

- [`basic_mlp`](examples/basic_mlp/) — Train an MLP and watch the loss decrease
- [`save_mlp`](examples/save_mlp/) — Train, serialize to JSON, deserialize, and verify output parity
- [`web_mlp`](examples/web_mlp/) — **Run the same MLP training in the browser** (WebAssembly)
- [`basic_autograd`](examples/basic_autograd/) — Forward + backward with `Variable`
- [`basic_tensor_ops`](examples/basic_tensor_ops/) — Tensor constructors and numeric ops

### Browser Demo

The same MoonBit training code runs in the browser via WebAssembly with zero source-level changes.

```bash
moon build --target wasm-gc
python3 -m http.server 8080
# Open http://localhost:8080/examples/web_mlp/index.html
```

Requires a wasm-gc capable browser: Chrome 119+, Firefox 120+, or Safari 18.2+.
See [`examples/web_mlp/`](examples/web_mlp/) for details.

## Module Structure

MbTorch is designed as a modular monorepo with clear boundaries between components:

```text
mbtorch/
├── core/
│   ├── tensor/     # Tensor types, constructors, and numeric ops
│   └── autograd/   # Reverse-mode autograd engine (Variable, backward)
├── nn/             # Neural network layers (Linear, ReLU)
├── optim/          # Optimization algorithms (SGD)
├── io/             # Model import/export: .mbt JSON (✅), ONNX, safetensors (planned)
├── examples/       # Working demos
└── tests/          # Unit and integration tests
```

### Module Responsibilities

- `core`  
  - Tensor types, numerical kernels, and automatic differentiation  
- `nn`  
  - High-level layers and model composition  
- `optim`  
  - Optimizers that operate on parameters produced by `nn`  
- `io`  
  - ONNX graph import, safetensors weight loading, `.mbt` model serialization/deserialization  
- `examples`  
  - End-to-end samples (browser demos, edge inference, fine-tuning flows)  
- `tests`  
  - TDD-oriented test suites for all modules  

The dependency graph is kept minimal and explicit to keep MbTorch lightweight and easy to embed.

## Development Style

MbTorch is developed with **Test-Driven Development (TDD)** and AI-assisted coding as first-class practices.

- New functionality starts with tests:  
  1. Write or refine tests that describe the expected behavior  
  2. Implement the minimal code needed to make tests pass  
  3. Refactor while keeping the test suite green  
- AI coding tools (e.g. Claude Code) are used primarily for implementation and refactoring, while humans focus on direction, design, and review.  
- Tests serve as executable documentation for contributors and users.

## Roadmap

### Phase 1: Core and TDD Foundation

Goal: Establish a minimal but solid core in MoonBit, with TDD and CI as the default workflow.

- Project layout and build/test tooling set up ✅
- Tensor type and core operations (add, mul, matmul, basic reductions) ✅
- Autograd engine (reverse-mode) for scalars, 1D, and 2D tensors ✅
- Basic neural network layers: `Linear`/`Dense`, common activations (ReLU, etc.) ✅
- Optimizers: SGD ✅ / Adam (remaining)
- Synthetic-data integration tests where training reduces loss over time ✅ (78 tests passing)
- JSON-based model serialization (`.mbt`-style serialize/deserialize) ✅

### Phase 2: Model Formats and I/O

Goal: Make MbTorch interoperable with existing model ecosystems and support its own native format.

- **ONNX import (subset)**  
  - Support for a targeted subset of operators needed for small MLPs, CNNs, and lightweight Transformers  
  - Verification tests comparing MbTorch outputs with reference runtimes  
- **safetensors import**  
  - Parse header (JSON) and binary tensor buffer, supporting common dtypes and shapes  
  - Combine ONNX graphs with safetensors weights to reconstruct pretrained models  
- **MbTorch-native format (`.mbt` JSON)**  
  - `serialize_model` / `deserialize_model` for JSON-based model I/O ✅
  - Binary `.mbt` format (metadata + packed tensor buffer) — planned
- User-facing I/O APIs:
  - `Model.from_onnx(...)` / `Model.from_safetensors(...)`
  - Binary `save_mbt(...)` / `load_mbt(...)`  

### Phase 3: Browser, Edge & Fine-Tuning UX

Goal: Deliver the core user experience: local, privacy-preserving fine-tuning and inference on browsers and edge devices.

- **WASM/Edge Runtime**  
  - Build and optimize a MbTorch runtime path targeting WebAssembly for browser and edge runtimes  
  - Provide a minimal browser demo (e.g. simple classifier or MLP)  
- **Lightweight Fine-Tuning**  
  - Support parameter-efficient tuning (e.g. LoRA/adapters) on top of imported models  
  - Partial training (freeze base model, train only additional parameters)  
- **Offline Optimization Flow (MVP)**  
  - Download a pretrained model (ONNX + safetensors) once  
  - Run short on-device fine-tuning on user data  
  - Save the personalized model in `.mbt` format to local storage (browser or file system)  
- **Docs and Examples**  
  - “Build a tiny MLP in MoonBit” tutorial  
  - “Import a PyTorch model via ONNX into MbTorch” guide  
  - Browser demo for local personalization and inference  

### Future Directions

Potential future extensions (subject to change):

- Additional layer types (CNNs, attention, RNNs)  
- Quantization and model compression for smaller footprints on edge devices  
- WebGPU or other accelerators when available in target environments  
- Federated learning-style workflows built on top of MbTorch’s on-device training  

## Installation

> NOTE: Installation and packaging details will be added once the project reaches a usable alpha. Integration with the MoonBit package ecosystem is planned.

## Contributing

Contributions are welcome!

- Use TDD: add or update tests for any new behavior.  
- Keep dependencies minimal and designs simple.  
- Optimize for readability and maintainability first; micro-optimizations can follow with benchmarks.  
- Open issues or discussions for design changes or larger features before implementation.

## License

MbTorch is licensed under the Apache License 2.0. See LICENSE for details.

## Acknowledgements

This project is built on top of the amazing MoonBit language and toolchain.  
I would like to thank the MoonBit team and the community for their continuous work on the compiler, libraries, and ecosystem, which made experiments like this framework possible.  
Any mistakes or limitations in this project are entirely my own.

***

**MbTorch** – Lightweight, privacy-first ML for the MoonBit and WebAssembly era.
