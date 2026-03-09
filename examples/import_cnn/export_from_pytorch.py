"""
Export a minimal CNN from PyTorch to ONNX + safetensors + expected_output.json.

Usage:
    cd examples/import_cnn
    python export_from_pytorch.py

Output:
    model.onnx            — ONNX graph (Conv2d + BN + ReLU + Flatten + Linear)
    model.safetensors     — Weight tensors (keys match ONNX initializer names)
    expected_output.json  — PyTorch inference output for verification
"""

import json
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from safetensors.torch import save_file

# --- Model definition ---
torch.manual_seed(42)

model = nn.Sequential(
    nn.Conv2d(1, 2, kernel_size=3, padding=1),  # 1ch -> 2ch, 3x3, pad=1
    nn.BatchNorm2d(2),
    nn.ReLU(),
    nn.Flatten(),             # [N, 2, 4, 4] -> [N, 32]
    nn.Linear(2 * 4 * 4, 3),  # 32 -> 3 classes
)

# Freeze BN (eval mode for consistent results)
model.eval()
print("Model created (eval mode for BN)")

# --- Test input: N=2, C=1, H=4, W=4 ---
test_input = torch.arange(1.0, 33.0).reshape(2, 1, 4, 4)

# --- 1. Export ONNX ---
out_dir = Path(__file__).parent
onnx_path = out_dir / "model.onnx"

torch.onnx.export(
    model,
    test_input,
    str(onnx_path),
    opset_version=17,
    input_names=["X"],
    output_names=["Y"],
    dynamic_axes=None,
    training=torch.onnx.TrainingMode.EVAL,
)
print(f"Exported ONNX to {onnx_path}")

# --- 2. Export safetensors (keys = ONNX initializer names) ---
onnx_model = onnx.load(str(onnx_path))
init_names = [init.name for init in onnx_model.graph.initializer]
print(f"ONNX initializer names: {init_names}")

state_dict = model.state_dict()
pytorch_keys = list(state_dict.keys())
assert len(pytorch_keys) == len(init_names), (
    f"Parameter count mismatch: PyTorch has {len(pytorch_keys)}, "
    f"ONNX has {len(init_names)}"
)

renamed = {}
for pt_key, onnx_name in zip(pytorch_keys, init_names):
    renamed[onnx_name] = state_dict[pt_key]
    print(f"  {pt_key} -> {onnx_name}  shape={list(state_dict[pt_key].shape)}")

st_path = out_dir / "model.safetensors"
save_file(renamed, str(st_path))
print(f"Exported safetensors to {st_path}")

# --- 3. Export expected output ---
with torch.no_grad():
    expected = model(test_input)

expected_list = expected.cpu().numpy().tolist()
json_path = out_dir / "expected_output.json"
with open(json_path, "w") as f:
    json.dump(expected_list, f)
print(f"Exported expected output to {json_path}")
print(f"Expected output shape: {list(expected.shape)}")
print(f"Expected output: {expected_list}")
