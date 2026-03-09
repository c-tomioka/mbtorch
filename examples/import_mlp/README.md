# Import MLP from PyTorch

Import a PyTorch-trained MLP into MbTorch via ONNX + safetensors and verify that the inference output matches.

## What it does

1. **Python** (`export_from_pytorch.py`): Defines a 2-layer MLP (`Linear(2,4) → ReLU → Linear(4,1)`), trains it briefly on synthetic data, then exports:
   - `model.onnx` — ONNX graph structure
   - `model.safetensors` — Weight tensors
   - `expected_output.json` — PyTorch inference output for a fixed test input
2. **MoonBit** (`main.mbt`): Reads the 3 files, reconstructs the model using `load_model_from_onnx_and_safetensors`, runs forward inference with the same test input, and compares element-wise against `expected_output.json`.

## Prerequisites

- Python 3.8+
- MoonBit toolchain (`moon`)

```bash
pip install torch safetensors onnx
```

## Run

```bash
# Step 1: Export from PyTorch
cd examples/import_mlp
python export_from_pytorch.py
cd ../..

# Step 2: Run MbTorch verification
moon run examples/import_mlp
```

## Expected output

```
Loaded 2 layers from ONNX + safetensors
Max absolute difference: ...
OK
```

`OK` means all output values match within 1e-5 absolute tolerance (float32 precision).
If `NG` is printed, the difference exceeds the tolerance threshold.

## Files

| File | Description |
|------|-------------|
| `export_from_pytorch.py` | Python script to train and export the model |
| `main.mbt` | MoonBit CLI that imports and verifies the model |
| `moon.pkg` | MoonBit package manifest |
| `model.onnx` | (generated) ONNX graph |
| `model.safetensors` | (generated) Weight data |
| `expected_output.json` | (generated) PyTorch reference output |
