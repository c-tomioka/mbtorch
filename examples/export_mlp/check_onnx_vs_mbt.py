"""
Verify MbTorch ONNX export against onnxruntime.

1. Runs `moon run examples/export_mlp` to generate model.onnx + mbtorch_output.json
2. Loads model.onnx with onnxruntime and runs inference
3. Compares MbTorch output vs ONNX Runtime output

Usage:
    cd <project_root>
    python examples/export_mlp/check_onnx_vs_mbt.py

Requirements:
    pip install onnxruntime numpy
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from onnxruntime import InferenceSession

# --- 1. Run MbTorch to export ONNX and save output ---
print("=== Step 1: Running MbTorch export ===")
result = subprocess.run(
    ["moon", "run", "examples/export_mlp"],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("ERROR: moon run failed", file=sys.stderr)
    print(result.stderr, file=sys.stderr)
    sys.exit(1)

# --- 2. Load MbTorch output ---
example_dir = Path(__file__).parent
mbt_output = json.loads((example_dir / "mbtorch_output.json").read_text())
mbt_array = np.array(mbt_output, dtype=np.float32)
print(f"MbTorch output:       {mbt_array.flatten()}")

# --- 3. Run ONNX Runtime inference ---
onnx_path = str(example_dir / "model.onnx")
session = InferenceSession(onnx_path)

# Same test input as main.mbt
test_input = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]], dtype=np.float32)

input_name = session.get_inputs()[0].name
ort_output = session.run(None, {input_name: test_input})[0]
print(f"ONNX Runtime output:  {ort_output.flatten()}")

# --- 4. Compare ---
diff = np.abs(mbt_array - ort_output)
max_diff = diff.max()
print(f"Max abs diff:         {max_diff:.2e}")

if max_diff < 1e-5:
    print("\nParity check PASSED!")
else:
    print(f"\nWARNING: max diff ({max_diff:.2e}) exceeds 1e-5 threshold")
    sys.exit(1)
