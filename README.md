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

MbTorch has completed **Phase 2** of its roadmap. Building on the Phase 1 foundation of tensors, autograd, MLP/CNN layers, and model I/O, Phase 2 added three major capabilities: ONNX export for all six layer types at opset 13 (ADR-0010), a DType extension with float16 and int8 storage that reduces model sizes by 50–75% (ADR-0011), and a Self-Attention MVP with full autograd support enabling Transformer-style model import, export, and training (ADR-0012). The autograd engine was extended to N-dimensional tensors with five new gradient functions. All 177 tests pass (wasm-gc build green). You can now define, train, save/restore, import, and export neural networks — including MLP, CNN, and attention-based models — entirely in MoonBit.

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
- **ONNX Export** — `export_onnx()` for sequential models (Linear, Conv2d, BatchNorm2d, Relu, Flatten, SelfAttention); opset 13 compatible; `export_onnx_with_dtype()` for float16 export
- **safetensors Import** — Binary parser for safetensors format; float32 and float16 tensors; JSON header + raw data layout
- **DType Extension** — `DType` enum (Float64/Float32/Float16/Int8); `Tensor::to_dtype()` conversion; `QuantParams` for int8 weight-only quantization; integrated with binary `.mbt`, ONNX export, and safetensors import
- **E2E Import** — `load_model_from_onnx_and_safetensors` for MLPs; `load_cnn_model_from_onnx_and_safetensors` for CNNs; `import_self_attention_from_onnx` for attention layers; PyTorch → ONNX + safetensors → MbTorch import with forward inference parity verified
- **Examples** — 7 working demos: `basic_tensor_ops`, `basic_autograd`, `basic_mlp`, `save_mlp`, `import_mlp` (PyTorch MLP import), `import_cnn` (PyTorch CNN import), `web_mlp` (browser demo)

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

### Phase 2: Model Formats and I/O ✅

Goal: Make MbTorch interoperable with existing model ecosystems and support its own native format.

- **ONNX import (Phase 1 subset)** ✅
  - Hand-written protobuf parser for ONNX ModelProto/GraphProto/TensorProto
  - Supported ops: Gemm, MatMul, Add, Relu (float32, 1D/2D tensors, sequential MLPs)
  - `parse_onnx` / `load_onnx_model` APIs
- **safetensors import (Phase 1 subset)** ✅
  - Binary parser: JSON header + raw float32/float16 tensor data
  - `parse_safetensors_header` / `load_safetensors` APIs
- **E2E import path** ✅
  - `load_model_from_onnx_and_safetensors`: ONNX structure + safetensors weights → Linear layers
  - PyTorch → torch.onnx.export + safetensors.save → MbTorch import → inference verified
- **MbTorch-native format (`.mbt` JSON)** ✅
  - `serialize_model` / `deserialize_model` for JSON-based model I/O
- **End-to-end PyTorch → MbTorch import demo** ✅
  - Export a trained PyTorch MLP to ONNX + safetensors in Python
  - Import it into MbTorch via `load_model_from_onnx_and_safetensors`
  - End-to-end parity verified in `examples/import_mlp` (max error ≈ 4.3e-7)
- **MbTorch-native binary format (`.mbt` binary)** ✅
  - `serialize_model_to_binary` / `deserialize_model_from_binary` APIs
  - float32 packed tensor buffer with JSON metadata; interconvertible with JSON `.mbt`
  - MVP scope: Linear/MLP, 1D/2D tensors
- **Conv2d + BatchNorm2d (inference-only)** ✅
  - `Conv2d` / `BatchNorm2d` layers and `Layer` enum for heterogeneous models
  - ONNX import: Conv, BatchNormalization, Relu, Flatten, Gemm → `load_cnn_model_from_onnx_and_safetensors`
  - Constraints: 2D conv, groups=1, dilations=[1,1], float32, NCHW
  - E2E parity verified in `examples/import_cnn`
- **ONNX export** ✅
  - `export_onnx()` / `export_onnx_with_dtype()` for sequential models (all 6 Layer types)
  - Opset 13 compatible; float32 and float16 export
- **DType extension (float16 / int8)** ✅
  - `DType` enum (Float64/Float32/Float16/Int8); `Tensor::to_dtype()` conversion
  - `QuantParams` for int8 weight-only quantization
  - Binary `.mbt` float16/int8 support; ONNX float16 export; safetensors float16 import
  - Precision verified: f16 MLP inference error < 1e-2, int8 weight-only error < 0.5
- **Self-Attention MVP** ✅
  - `SelfAttention` struct (Wq/Wk/Wv/Wo + num_heads); single-head and multi-head
  - Tensor ops: exp, softmax, batched_matmul_2d, transpose_last2
  - Autograd: 5 new GradFn (Scale, Softmax, BatchedMatmul, TransposeLast2, Reshape); N-D tensor support
  - ONNX import: 9-node pattern match (Gemm×3+Transpose+MatMul+Div+Softmax+MatMul+Gemm)
  - ONNX export: 9-node subgraph (opset 13 compatible)
- Pooling layers (MaxPool2d, AvgPool2d) — planned
- Binary `.mbt` checksum — planned

> Phase 2 total: **177 tests passing**. Design decisions documented in [`docs/adr/`](docs/adr/) (ADR-0008 through ADR-0012).

### Phase 3: Browser, Edge & Fine-Tuning UX

Goal: Deliver the core user experience: local, privacy-preserving fine-tuning and inference on browsers and edge devices.

- **WASM/Edge Runtime**
  - Build and optimize a MbTorch runtime path targeting WebAssembly for browser and edge runtimes
  - Browser demo: `web_mlp` with 6-variant comparison (SGD/Adam × ReLU/tanh/sigmoid) ✅
  - Browser demos for CNN and attention model inference — planned
- **Conv2d Training**
  - Backward pass and gradient computation for Conv2d/BatchNorm2d
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

- Additional layer types: pooling (MaxPool2d, AvgPool2d), RNNs, LayerNorm, TransformerBlock
- Extended Attention: Cross-Attention, causal mask, KV cache, Grouped Query Attention
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
