"""
Export a minimal MLP from PyTorch to ONNX + safetensors + expected_output.json.

Usage:
    cd examples/import_mlp
    python export_from_pytorch.py

Output:
    model.onnx            — ONNX graph (Gemm + Relu + Gemm)
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
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)

# --- Quick training on synthetic data (y ≈ 2*x1 + 3*x2) ---
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x_train = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [3.0, 2.0]])
y_train = torch.tensor([[5.0], [7.0], [8.0], [12.0]])

for _ in range(200):
    pred = model(x_train)
    loss = ((pred - y_train) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
print(f"Training done. Final loss: {loss.item():.6f}")

# --- Test input (same values hardcoded in main.mbt) ---
test_input = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])

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
)
print(f"Exported ONNX to {onnx_path}")

# --- 2. Export safetensors (keys = ONNX initializer names) ---
# PyTorch state_dict keys: "0.weight", "0.bias", "2.weight", "2.bias"
# ONNX initializer names vary by export; read them from the ONNX model.
onnx_model = onnx.load(str(onnx_path))
init_names = [init.name for init in onnx_model.graph.initializer]
print(f"ONNX initializer names: {init_names}")

# Map PyTorch state_dict keys to ONNX initializer names by order.
# PyTorch Sequential exports parameters in layer order:
# [layer0.weight, layer0.bias, layer2.weight, layer2.bias]
# ONNX initializers follow the same order.
state_dict = model.state_dict()
pytorch_keys = list(state_dict.keys())
assert len(pytorch_keys) == len(init_names), (
    f"Parameter count mismatch: PyTorch has {len(pytorch_keys)}, "
    f"ONNX has {len(init_names)}"
)

renamed = {}
for pt_key, onnx_name in zip(pytorch_keys, init_names):
    renamed[onnx_name] = state_dict[pt_key]
    print(f"  {pt_key} -> {onnx_name}")

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
print(f"Expected output: {expected_list}")
