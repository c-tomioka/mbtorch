"""
Verify MbTorch CNN ONNX export against onnxruntime.

1. Runs `moon run examples/export_cnn` to generate model.onnx + outputs
2. Loads model.onnx with onnxruntime and runs inference with the same input
3. Compares MbTorch output vs ONNX Runtime output

Usage:
    cd <project_root>
    python3 examples/export_cnn/check_onnx_vs_mbt.py

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
print("=== Step 1: Running MbTorch CNN export ===")
result = subprocess.run(
    ["moon", "run", "examples/export_cnn"],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("ERROR: moon run failed", file=sys.stderr)
    print(result.stderr, file=sys.stderr)
    sys.exit(1)

# --- 2. Load MbTorch output and input ---
example_dir = Path(__file__).parent
mbt_output = np.array(
    json.loads((example_dir / "mbtorch_output.json").read_text()),
    dtype=np.float32,
)
input_flat = json.loads((example_dir / "input.json").read_text())
test_input = np.array(input_flat, dtype=np.float32).reshape(1, 1, 4, 4)

print(f"Input shape:          {test_input.shape}")
print(f"MbTorch output:       {mbt_output}")

# --- 3. Run ONNX Runtime inference ---
onnx_path = str(example_dir / "model.onnx")
session = InferenceSession(onnx_path)

input_name = session.get_inputs()[0].name
ort_output = session.run(None, {input_name: test_input})[0].flatten()
print(f"ONNX Runtime output:  {ort_output}")

# --- 4. Compare ---
diff = np.abs(mbt_output - ort_output)
max_diff = diff.max()
print(f"Max abs diff:         {max_diff:.2e}")

if max_diff < 1e-5:
    print("\nParity check PASSED!")
else:
    print(f"\nWARNING: max diff ({max_diff:.2e}) exceeds 1e-5 threshold")
    sys.exit(1)
