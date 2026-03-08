# MbTorch

Lightweight AI/ML framework in MoonBit. Train, fine-tune, and run neural networks directly in the browser and on edge devices via WebAssembly. Privacy-first, serverless ML for everyone.[1][2]

## Overview

**MbTorch** is an open-source machine learning framework built in MoonBit, designed for lightweight models and on-device adaptation. It focuses on enabling users to build small-to-medium neural networks from scratch in MoonBit, import existing pretrained models (via ONNX and safetensors), and run training and inference efficiently on browsers and edge devices through WebAssembly.[2][3][1]

Unlike traditional ML frameworks that assume powerful servers and large-scale training, MbTorch embraces constrained environments and offline usage. The goal is to let users obtain an optimized model for their own data and device, entirely on-device and without sending private data to the cloud.[4][1]

### Why MbTorch?

Most AI frameworks prioritize massive, server-side models. MbTorch takes a different approach:

- **Train on-device**: Keep sensitive data local; no mandatory cloud backend.[5]
- **Lightweight by design**: Target practical models suitable for CPU/WASM, rather than giant distributed training setups.[6][2]
- **WebAssembly-first**: Compile MoonBit to WASM and run anywhere a WebAssembly runtime is available (browsers, edge runtimes, serverless).[7][1]
- **Privacy-preserving**: Support personal fine-tuning and adaptation without uploading raw data.  
- **Interoperable**: Import models via ONNX and safetensors to reuse existing ecosystems and tooling.[3][8]
- **MoonBit native**: Leverage MoonBit’s AI-friendly, WASM-oriented design, small binaries, and fast compilation for edge AI workflows.[1][4]

### Target Use Cases

MbTorch focuses on small and mid-sized neural networks and edge-centric tasks, for example:

- Handwritten digit or character recognition in the browser  
- Sensor data anomaly detection on IoT devices  
- Simple recommendation or ranking systems on-device  
- Local personalization and LoRA-style fine-tuning of imported models  
- Educational ML projects and interactive browser demos  

## Features

Implemented and planned features:

- **Tensor Operations** ✅: Multi-dimensional tensors with element-wise ops (add, mul), matrix multiplication (matmul), transpose, sum, and more
- **Automatic Differentiation** ✅: Reverse-mode autograd engine with `Variable` wrapper, supporting add/mul/matmul/sum backward passes
- **Neural Network Layers** (planned): Basic layers such as dense/linear layers and activations
- **Optimizers** (planned): Common optimizers like SGD and Adam
- **Data Utilities** (planned): Mini-batching helpers and simple preprocessing utilities
- **Model I/O** (planned):
  - Import from **ONNX** for graph structure and operators[9][3]
  - Import from **safetensors** for efficient, safe weight loading[8][10]
  - Export and load a **MbTorch-native format** (e.g. `.mbt`) for compact MoonBit-native serialization
- **Edge & Browser Runtime** (planned): WASM-targeted execution paths for inference and lightweight fine-tuning on edge and browser environments[7][1]

## Quick Start

> MbTorch is in early development. APIs are experimental and subject to change.

```moonbit
// Forward + backward with autograd (this code runs today!)
fn main {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([1.0, 2.0, 3.0]),
    requires_grad=true,
  )
  let w = @autograd.Variable::new(
    @tensor.Tensor::from_array([0.5, -1.0, 2.0]),
    requires_grad=true,
  )

  // loss = sum(x * w)
  let loss = x.mul(w).sum()
  loss.backward()

  println(x.grad().unwrap())  // Tensor(shape=[3], data=[0.5, -1, 2])
  println(w.grad().unwrap())  // Tensor(shape=[3], data=[1, 2, 3])
}
```

See `examples/` for more working demos. Neural network layers (`nn`), optimizers (`optim`), and model I/O (`io`) are under active development.

## Module Structure

MbTorch is designed as a modular monorepo with clear boundaries between components:

```text
mbtorch/
├── core/
│   ├── tensor/     # Tensor types, constructors, and numeric ops
│   └── autograd/   # Reverse-mode autograd engine (Variable, backward)
├── nn/             # Neural network layers (planned)
├── optim/          # Optimization algorithms (planned)
├── io/             # Model import/export: ONNX, safetensors, .mbt (planned)
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
  - ONNX graph import, safetensors weight loading, `.mbt` model serialization/deserialization[10][9]
- `examples`  
  - End-to-end samples (browser demos, edge inference, fine-tuning flows)  
- `tests`  
  - TDD-oriented test suites for all modules  

The dependency graph is kept minimal and explicit to keep MbTorch lightweight and easy to embed.

## Development Style

MbTorch is developed with **Test-Driven Development (TDD)** and AI-assisted coding as first-class practices.[11][12]

- New functionality starts with tests:  
  1. Write or refine tests that describe the expected behavior  
  2. Implement the minimal code needed to make tests pass  
  3. Refactor while keeping the test suite green  
- AI coding tools (e.g. Claude Code) are used primarily for implementation and refactoring, while humans focus on direction, design, and review.  
- Tests serve as executable documentation for contributors and users.

## Roadmap

### Phase 1: Core and TDD Foundation (0–3 months)

Goal: Establish a minimal but solid core in MoonBit, with TDD and CI as the default workflow.

- Project layout and build/test tooling set up  
- Tensor type and core operations (add, mul, matmul, basic reductions)  
- Autograd engine (reverse-mode) for scalars, 1D, and 2D tensors  
- Basic neural network layers: `Linear`/`Dense`, common activations (ReLU, etc.)  
- Optimizers: SGD and Adam  
- Synthetic-data integration tests where training reduces loss over time  

### Phase 2: Model Formats and I/O (3–6 months)

Goal: Make MbTorch interoperable with existing model ecosystems and support its own native format.

- **ONNX import (subset)**  
  - Support for a targeted subset of operators needed for small MLPs, CNNs, and lightweight Transformers[3][9]
  - Verification tests comparing MbTorch outputs with reference runtimes  
- **safetensors import**  
  - Parse header (JSON) and binary tensor buffer, supporting common dtypes and shapes[8][10]
  - Combine ONNX graphs with safetensors weights to reconstruct pretrained models  
- **MbTorch-native format (`.mbt`, tentative)**  
  - Design a simple metadata + binary format (e.g. JSON metadata + packed tensor buffer)  
  - Implement `save_mbt` / `load_mbt` for MoonBit-native models  
- User-facing I/O APIs:  
  - `Model.from_onnx(...)` / `Model.from_safetensors(...)`  
  - `model.save_mbt(...)` / `Model.load_mbt(...)`  

### Phase 3: Browser, Edge & Fine-Tuning UX (6–12 months)

Goal: Deliver the core user experience: local, privacy-preserving fine-tuning and inference on browsers and edge devices.

- **WASM/Edge Runtime**  
  - Build and optimize a MbTorch runtime path targeting WebAssembly for browser and edge runtimes[13][1][7]
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
  - “Import a PyTorch model via ONNX into MbTorch” guide[14][15]
  - Browser demo for local personalization and inference  

### Future Directions

Potential future extensions (subject to change):

- Additional layer types (CNNs, attention, RNNs)  
- Quantization and model compression for smaller footprints on edge devices  
- WebGPU or other accelerators when available in target environments  
- Federated learning-style workflows built on top of MbTorch’s on-device training  

## Installation

> NOTE: Installation and packaging details will be added once the project reaches a usable alpha. Integration with the MoonBit package ecosystem is planned.[16][1]

## Contributing

Contributions are welcome!

- Use TDD: add or update tests for any new behavior.  
- Keep dependencies minimal and designs simple.  
- Optimize for readability and maintainability first; micro-optimizations can follow with benchmarks.  
- Open issues or discussions for design changes or larger features before implementation.

## License

MbTorch is intended to be released under a permissive open-source license (e.g. Apache 2.0). License details will be finalized and documented in `LICENSE`.

***

**MbTorch** – Lightweight, privacy-first ML for the MoonBit and WebAssembly era.
