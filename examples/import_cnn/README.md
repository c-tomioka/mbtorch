# import_cnn

Import a small CNN (Conv2d → BatchNorm2d → ReLU → Flatten → Linear) from PyTorch via ONNX + safetensors, then run forward inference in MbTorch and compare against PyTorch expected output.

## Prerequisites

```bash
pip install torch onnx safetensors
```

## Export model from PyTorch

```bash
cd examples/import_cnn
python export_from_pytorch.py
```

This generates:
- `model.onnx` — ONNX graph
- `model.safetensors` — Weight tensors
- `expected_output.json` — PyTorch inference output

## Run MbTorch inference

```bash
moon run examples/import_cnn
```

## Expected output

```
Loaded 5 layers from ONNX + safetensors
  [0] ...  (Conv)
  [1] ...  (BatchNormalization)
  [2] ...  (Relu)
  [3] ...  (Flatten)
  [4] ...  (Gemm/Linear)
Output shape: [2, 3]
Max absolute difference: ...
OK
```
