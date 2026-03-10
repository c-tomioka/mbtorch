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

MbTorch focuses on small and mid-sized neural networks (MLP, CNN, Self-Attention) and edge-centric tasks, for example:

- Handwritten digit or character recognition in the browser
- Sensor data anomaly detection on IoT devices
- Small Transformer-based models (e.g. tiny BERT/GPT) for on-device NLP
- Simple recommendation or ranking systems on-device
- Local personalization and LoRA-style fine-tuning of imported models
- Educational ML projects and interactive browser demos

## Project Status

MbTorch has completed **Phase 2** of its roadmap. Building on the Phase 1 foundation of tensors, autograd, MLP/CNN layers, and model I/O, Phase 2 added three major capabilities: ONNX export for all six layer types at opset 13 (ADR-0010), a DType extension with float16 and int8 storage that reduces model sizes by 50–75% (ADR-0011), and a Self-Attention MVP with full autograd support enabling Transformer-style model import, export, and training (ADR-0012). The autograd engine was extended to N-dimensional tensors with five new gradient functions. All 189 tests pass (wasm-gc build green). ONNX export correctness is verified against onnxruntime for both MLP and CNN models. You can now define, train, save/restore, import, and export neural networks — including MLP, CNN, and attention-based models — entirely in MoonBit.

**Next up:** Conv2d training (backward pass), lightweight fine-tuning (LoRA/adapters), and browser demos for CNN/attention inference.

## Features

### Implemented ✅

- **Tensor Operations** — Multi-dimensional tensors with add, sub, mul, div, scale, matmul, transpose, sum, sqrt, reshape, conv2d, exp, softmax, batched_matmul_2d, transpose_last2; constructors for scalar, 1D, 2D, N-D, zeros, ones, rand
- **Automatic Differentiation** — Reverse-mode autograd engine (`Variable`, `backward`) with gradient support for add/sub/mul/matmul/sum/relu/tanh/sigmoid/expand_rows/scale/softmax/batched_matmul_2d/transpose_last2/reshape; N-D tensor support; includes `grad_check` for numerical verification
- **Neural Network Layers** — `Linear` layer with forward/from_tensors; `Conv2d` (inference-only, NCHW, groups=1); `BatchNorm2d` (inference-only, frozen running stats); `SelfAttention` (single/multi-head, autograd-enabled); `Layer` enum (`Linear | Conv2d | BatchNorm2d | Relu | Flatten | SelfAttention`) for heterogeneous model composition; composable MLPs with ReLU, tanh, and sigmoid activations
- **Optimizers** — `SGD` and `Adam` optimizers with `new`, `step()`, `zero_grad()`
- **Model I/O** — Serialize/deserialize models to `.mbt`-style JSON via `serialize_model` / `deserialize_model`; roundtrip-tested with trained models
- **Binary `.mbt` Format** — Compact binary serialization (`serialize_model_to_binary` / `deserialize_model_from_binary`); JSON metadata header + packed tensor buffer; float32/float16/int8 dtype support with `serialize_model_to_binary_with_dtype`; interconvertible with JSON `.mbt`
- **ONNX Import** — Hand-written protobuf parser; MLP support (Gemm/MatMul/Add/Relu → Linear); CNN support (Conv/BatchNormalization/Relu/Flatten/Gemm → Layer enum); SelfAttention support (9-node Gemm×3+Transpose+MatMul+Div+Softmax+MatMul+Gemm pattern); float32 and float16 tensors
- **ONNX Export** — `export_onnx()` for sequential models (Linear, Conv2d, BatchNorm2d, Relu, Flatten, SelfAttention); opset 13 compatible; onnxruntime parity verified for MLP and CNN; `export_onnx_with_dtype()` for float16 export
- **safetensors Import** — Binary parser for safetensors format; float32 and float16 tensors; JSON header + raw data layout
- **DType Extension** — `DType` enum (Float64/Float32/Float16/Int8); `Tensor::to_dtype()` conversion; `QuantParams` for int8 weight-only quantization; integrated with binary `.mbt`, ONNX export, and safetensors import
- **E2E Import** — `load_model_from_onnx_and_safetensors` for MLPs; `load_cnn_model_from_onnx_and_safetensors` for CNNs; `import_self_attention_from_onnx` for attention layers; PyTorch → ONNX + safetensors → MbTorch import with forward inference parity verified
- **Examples** — 9 working demos: `basic_tensor_ops`, `basic_autograd`, `basic_mlp`, `save_mlp`, `import_mlp` (PyTorch MLP import), `import_cnn` (PyTorch CNN import), `export_mlp` (ONNX export + onnxruntime verification), `export_cnn` (CNN ONNX export + onnxruntime verification), `web_mlp` (browser demo)

### Planned

- **Pooling Layers** — MaxPool2d, AvgPool2d
- **Conv2d Training** — Backward pass and gradient computation for Conv2d/BatchNorm2d
- **Lightweight Fine-Tuning** — LoRA/adapter-based parameter-efficient tuning on imported models
- **Data Utilities** — Mini-batching helpers and preprocessing
- **Edge & Browser Demos** — CNN inference and attention model demos in the browser

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
- [`save_mlp`](examples/save_mlp/) — Train, serialize to JSON and binary `.mbt`, and verify output parity
- [`import_mlp`](examples/import_mlp/) — Import a PyTorch MLP via ONNX + safetensors and verify inference parity
- [`import_cnn`](examples/import_cnn/) — Import a PyTorch CNN (Conv2d → BN → ReLU → Flatten → Linear) via ONNX + safetensors
- [`export_mlp`](examples/export_mlp/) — Train an MLP, export to ONNX, verify with onnxruntime (parity < 1e-6)
- [`export_cnn`](examples/export_cnn/) — Build a CNN (Conv2d → BN → ReLU → Flatten → Linear), export to ONNX, verify with onnxruntime (parity < 1e-7)
- [`web_mlp`](examples/web_mlp/) — **Run the same MLP training in the browser** (WebAssembly)
- [`basic_autograd`](examples/basic_autograd/) — Forward + backward with `Variable`
- [`basic_tensor_ops`](examples/basic_tensor_ops/) — Tensor constructors and numeric ops

### Browser Demo

MbTorch runs natively in the browser via WebAssembly — the same MoonBit training code compiles to wasm-gc and executes with zero source changes. The [`web_mlp`](examples/web_mlp/) demo trains 6 MLP variants in-browser and visualizes their loss curves in an interactive chart, letting you compare optimizers and activations at a glance.

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
├── nn/             # Neural network layers (Linear, Conv2d, BatchNorm2d, SelfAttention, Layer enum)
├── optim/          # Optimization algorithms (SGD, Adam)
├── io/             # Model I/O: .mbt JSON/binary, ONNX import/export, safetensors import
├── examples/       # Working demos
└── tests/          # Unit and integration tests
```

### Module Responsibilities

- `core`  
  - Tensor types, numerical kernels, and automatic differentiation  
- `nn`
  - High-level layers (Linear, Conv2d, BatchNorm2d, SelfAttention), `Layer` enum for heterogeneous models, and activation dispatching
- `optim`  
  - Optimizers that operate on parameters produced by `nn`  
- `io`  
  - ONNX graph import/export, safetensors weight loading, `.mbt` model serialization/deserialization, DType conversion  
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

### Phase 1: Core and TDD Foundation ✅

Goal: Establish a minimal but solid core in MoonBit, with TDD and CI as the default workflow.

- Project layout and build/test tooling set up ✅
- Tensor type and core operations (add, mul, div, sqrt, matmul, basic reductions) ✅
- Autograd engine (reverse-mode) for scalars, 1D, and 2D tensors ✅
- Activation functions: ReLU, tanh, sigmoid (forward + backward) ✅
- Basic neural network layers: `Linear`/`Dense`, composable MLPs ✅
- Optimizers: SGD ✅ / Adam ✅
- Synthetic-data integration tests where training reduces loss over time ✅ (106 tests passing)
- JSON-based model serialization (`.mbt`-style serialize/deserialize) ✅

### Phase 2: Model Formats, ONNX & Attention ✅

Goal: Make MbTorch interoperable with existing model ecosystems, support its own native format, and cover MLP/CNN/Attention-style models end-to-end.

- **ONNX import (MLP subset)** ✅  
  - Hand-written protobuf parser for ONNX ModelProto/GraphProto/TensorProto  
  - Supported ops: Gemm, MatMul, Add, Relu (float32, 1D/2D tensors, sequential MLPs)  
  - `parse_onnx` / `load_onnx_model` APIs  

- **safetensors import (float32/float16)** ✅  
  - Binary parser: JSON header + raw float32/float16 tensor data  
  - `parse_safetensors_header` / `load_safetensors` APIs  

- **E2E import path (MLP)** ✅  
  - `load_model_from_onnx_and_safetensors`: ONNX structure + safetensors weights → `Linear` layers  
  - PyTorch → `torch.onnx.export` + `safetensors.save` → MbTorch import → inference parity verified  

- **MbTorch-native format (`.mbt` JSON)** ✅  
  - `serialize_model` / `deserialize_model` for JSON-based model I/O  

- **MbTorch-native binary format (`.mbt` binary)** ✅  
  - `serialize_model_to_binary` / `deserialize_model_from_binary` APIs  
  - Layout: `magic "MBTM"` + version + metadata_size + JSON metadata + packed tensor buffer  
  - DType-aware: float32 / float16 / int8 storage  
  - Interconvertible with JSON `.mbt`; v1 binary remains backward compatible  

- **Conv2d + BatchNorm2d (inference-only)** ✅  
  - `Conv2d` / `BatchNorm2d` layers and `Layer` enum for heterogeneous models  
  - ONNX import: Conv, BatchNormalization, Relu, Flatten, Gemm → `load_cnn_model_from_onnx_and_safetensors`  
  - Constraints: 2D conv, groups=1, dilations=[1,1], NCHW, float32  

- **ONNX export (Linear/CNN/Attention)** ✅  
  - `export_onnx()` for sequential models: `Linear`, `Conv2d`, `BatchNorm2d`, `Relu`, `Flatten`, `SelfAttention`  
  - Opset 13 compatible; float32  
  - `export_onnx_with_dtype()` for float16 export  

- **DType extension (float16 / int8 storage)** ✅  
  - `DType` enum (Float64/Float32/Float16/Int8) and `QuantParams` for int8 weight-only quantization  
  - `Tensor::to_dtype()` conversion between f32/f16/int8 (storage)  
  - Integrated with binary `.mbt`, ONNX export, safetensors import  
  - Precision verified: f16 MLP inference error < 1e-2; int8 weight-only Linear error < 0.5  

- **Self-Attention MVP** ✅  
  - `SelfAttention` struct (Wq/Wk/Wv/Wo + num_heads), autograd-enabled  
  - Core ops: `exp`, `softmax`, `batched_matmul_2d`, `transpose_last2`, `reshape`  
  - Autograd: scale/softmax/batched_matmul_2d/transpose_last2/reshape grad functions; N-D tensor support  
  - ONNX import: 9-node MatMul + Softmax self-attention pattern (Gemm×3 + Transpose + MatMul + Div + Softmax + MatMul + Gemm) → `SelfAttention`  
  - ONNX export: same 9-node subgraph (opset 13 compatible)  

> Phase 2 total: **189 tests passing** (including CNN, dtype, Self-Attention, and ONNX attribute/roundtrip verification).  
> Design decisions documented in [`docs/adr/`](docs/adr/) (ADR-0008 through ADR-0012).

### Phase 3: Browser, Edge & Fine-Tuning UX

Goal: Deliver the core user experience: local, privacy-preserving fine-tuning and inference on browsers and edge devices, built on top of the Phase 1–2 core.

- **WASM/Edge Runtime UX**  
  - Keep MbTorch’s WASM path efficient and small for browser and edge runtimes  
  - Existing demo: `web_mlp` trains 6 MLP variants entirely in-browser (SGD/Adam × ReLU/tanh/sigmoid) ✅  
  - Add browser demos for:
    - CNN inference (e.g. small ConvNet for digit/character recognition)  
    - Self-Attention / tiny Transformer inference on toy NLP tasks  

- **Conv2d & CNN Training**  
  - Implement backward pass and gradients for `Conv2d` / `BatchNorm2d`  
  - Extend autograd tests to cover CNN training (loss decreasing over time)  
  - Provide an example that trains a small CNN end-to-end in MoonBit (CLI)  

- **Lightweight Fine-Tuning & Personalization**  
  - Add parameter-freeze APIs and utilities (selectively train subsets of layers)  
  - Introduce LoRA/adapter-style layers for parameter-efficient fine-tuning on imported models  
  - Provide “on-device fine-tuning” flows:
    - Import ONNX + safetensors model → attach adapters → fine-tune only adapter parameters  
    - Save personalized variants as `.mbt` (JSON/binary)  

- **Data Utilities & Training UX**  
  - Mini-batch helpers (simple Dataset/DataLoader-style iteration)  
  - Common preprocessing utilities (normalization, standardization, basic augmentations)  
  - Lightweight training helpers:
    - Learning-rate schedulers (step, cosine, etc.)  
    - Gradient clipping and weight decay helpers  
    - Simple hooks for logging loss/metrics  

- **Docs and Examples**  
  - “Build a tiny MLP in MoonBit” tutorial  
  - “Import a PyTorch model via ONNX into MbTorch” guide (MLP/CNN/Attention)  
  - Browser demo and guide for local personalization and inference (fine-tune-once, run offline)

### Future Directions

Potential future extensions (subject to change):

- **Layers & Architectures**  
  - Pooling layers (MaxPool2d, AvgPool2d)  
  - LayerNorm, GroupNorm, InstanceNorm  
  - RNN/LSTM/GRU and time-series friendly layers  
  - Higher-level `TransformerBlock` (Self-Attention + MLP + LayerNorm + residuals)  
  - Extended Attention:
    - Cross-Attention, causal masks, KV cache  
    - Grouped Query / Multi-Query Attention  

- **Quantization & Precision**  
  - Native float16 execution paths and mixed-precision training  
  - More advanced int8 quantization:
    - Activation quantization, per-channel scales, QAT support  
  - Automatic quantization/dequantization passes and tooling for `.mbt` ↔ ONNX  

- **Data Pipeline & Orchestration**  
  - Richer Dataset/DataLoader abstractions  
  - Built-in logging, checkpointing, and early-stopping helpers  
  - Simple experiment tracking for edge/browsers (e.g. local storage-based histories)  

- **Edge Performance & Hardware Acceleration**  
  - WebGPU or other accelerators when available in target environments  
  - Platform-specific optimizations for common operations on constrained devices  

- **Federated / Collaborative Workflows**  
  - Federated-learning-style workflows built on MbTorch’s on-device training model  
  - Tools for aggregating `.mbt` updates across clients while preserving privacy  

- **Ecosystem & Tooling**  
  - CLI utilities to convert between `.mbt` and ONNX/safetensors variants  
  - More comprehensive E2E tests and benchmarks (including small Transformer models)  
  - Deeper integrations with MoonBit ecosystem packages and tooling

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
