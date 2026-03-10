# MbTorch Model Formats

MbTorch supports four model serialization formats for different use cases.

## Overview

| Format | Direction | DTypes | Use Case |
|--------|-----------|--------|----------|
| JSON `.mbt` | save / load | Float64 | Debugging, human-readable inspection |
| Binary `.mbt` | save / load | Float32 / Float16 / Int8 | Production, edge deployment, small footprint |
| ONNX | import / export | Float32 / Float16 | Interop with PyTorch, onnxruntime, other frameworks |
| safetensors | import only | Float32 / Float16 | Weight loading from HuggingFace ecosystem |

## JSON `.mbt` Format

Human-readable JSON serialization of MbTorch models.

- **Structure**: Array of `(name, tensor)` pairs serialized as nested JSON arrays of Double values
- **APIs**: `serialize_model()` / `deserialize_model()`
- **DType**: Float64 only (all values stored as JSON numbers)
- **Limitations**: Larger file size compared to binary; no dtype metadata; no quantization support
- **Use case**: Debugging, manual inspection, quick prototyping

**Reference**: [ADR-0005](adr/0005-mbt-json-format.md)

## Binary `.mbt` Format

Compact binary format optimized for storage and edge deployment.

### Wire Format

```
Offset  Size     Field
0       4        Magic bytes: "MBTO"
4       4        Version: UInt32 (currently 1)
8       4        Metadata size: UInt32 (byte length of JSON metadata)
12      N        JSON metadata string (layer names, shapes, dtypes, quant_params)
12+N    M        Packed tensor data buffer
```

### DType Support

| DType | Bytes/element | Notes |
|-------|--------------|-------|
| Float32 | 4 | Default; little-endian IEEE 754 |
| Float16 | 2 | IEEE 754 half-precision; stored compactly, computed in Double |
| Int8 | 1 | Weight-only quantization; requires `scale` and `zero_point` in metadata |

### APIs

- `serialize_model_to_binary(layers)` — Float32 default
- `serialize_model_to_binary_with_dtype(layers, dtype)` — Explicit dtype (Float16/Int8)
- `deserialize_model_from_binary(bytes)` — Auto-detects dtype from metadata

### Backward Compatibility

- Files without a `dtype` field in metadata default to Float32
- Version 1 files are always readable by the current parser

**Reference**: [ADR-0008](adr/0008-binary-mbt-format-mvp.md), [ADR-0011](adr/0011-dtype-extension-mvp.md)

## ONNX Support

ONNX (Open Neural Network Exchange) for interoperability with PyTorch and other frameworks.

- **Opset version**: 13 (widely supported by PyTorch export and onnxruntime)
- **Direction**: Import and export

### Import APIs

| API | Input | Output | Model Type |
|-----|-------|--------|------------|
| `load_model_from_onnx_and_safetensors` | ONNX + safetensors bytes | `Array[(String, Linear)]` | MLP |
| `load_cnn_model_from_onnx_and_safetensors` | ONNX + safetensors bytes | `Array[(String, Layer)]` | CNN |
| `import_self_attention_from_onnx` | ONNX bytes | `SelfAttention` | Attention |

### Export APIs

| API | Input | Output |
|-----|-------|--------|
| `export_onnx(layers, input_shape)` | `Array[Layer]` + shape | ONNX bytes (Float32) |
| `export_onnx_with_dtype(layers, input_shape, dtype)` | `Array[Layer]` + shape + dtype | ONNX bytes (Float16) |

### Operator Support Matrix

| MbTorch Layer | ONNX Import Ops | ONNX Export Ops |
|---------------|-----------------|-----------------|
| Linear | Gemm (transB) / MatMul+Add | Gemm (transB=1) |
| Conv2d | Conv | Conv |
| BatchNorm2d | BatchNormalization | BatchNormalization |
| Relu | Relu | Relu |
| Flatten | Flatten | Flatten (axis=1) |
| SelfAttention | 9-node pattern* | 9-node subgraph* |

*SelfAttention ONNX pattern: `Gemm(Q) + Gemm(K) + Gemm(V) + Transpose + MatMul + Div + Softmax + MatMul + Gemm(O)`

### Constraints

- Sequential models only (no arbitrary DAG graphs)
- SelfAttention import validates Q/K/V share the same input (self-attention only, no cross-attention)
- Conv2d: groups=1, dilations=[1,1] only
- Int8 is not supported in ONNX (use binary `.mbt` for int8 models)

**Reference**: [ADR-0007](adr/0007-phase1-onnx-safetensors.md), [ADR-0009](adr/0009-conv2d-batchnorm-mvp.md), [ADR-0010](adr/0010-onnx-export-mvp.md), [ADR-0012](adr/0012-attention-ops-mvp.md)

## safetensors Support

Import-only support for the [safetensors](https://github.com/huggingface/safetensors) format used by HuggingFace.

- **Structure**: 8-byte header length (little-endian) + JSON header + raw tensor data
- **APIs**: `parse_safetensors_header()` / `load_safetensors()`
- **DTypes**: Float32 (F32) and Float16 (F16)
- **Usage**: Combined with ONNX for end-to-end import — ONNX provides graph structure, safetensors provides weight tensors

**Reference**: [ADR-0007](adr/0007-phase1-onnx-safetensors.md)

## DType and Quantization

MbTorch supports multiple data types for storage efficiency while performing all computation in Double internally.

### DType Enum

| DType | Storage Size | Description |
|-------|-------------|-------------|
| Float64 | 8 bytes | Full precision; used in JSON `.mbt` and internal computation |
| Float32 | 4 bytes | Standard ML precision; default for binary `.mbt` and ONNX |
| Float16 | 2 bytes | Half precision; ~50% size reduction with minimal accuracy loss |
| Int8 | 1 byte | Weight-only quantization; ~75% size reduction |

### APIs

- `Tensor::to_dtype(target_dtype)` — Convert between dtypes
- `QuantParams { scale: Double, zero_point: Int }` — Parameters for int8 quantization

### Precision Benchmarks

| Scenario | Metric | Threshold |
|----------|--------|-----------|
| Float16 roundtrip | \|original - recovered\| | < 1e-3 |
| Int8 roundtrip | relative error | < 5% |
| MLP float16 inference | absolute error | < 1e-2 |
| Linear int8 inference | absolute error | < 0.5 |

**Reference**: [ADR-0011](adr/0011-dtype-extension-mvp.md)
