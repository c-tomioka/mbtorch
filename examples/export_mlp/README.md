# export_mlp — MbTorch ONNX Export Demo

Train a small MLP in MbTorch and export it to ONNX. Then verify the exported model with onnxruntime and visualize it with Netron.

## What it does

1. Defines a 2-layer MLP: `Linear(2,4) -> ReLU -> Linear(4,1)`
2. Trains on synthetic data (`y = 2*x1 + 3*x2`) for 50 steps
3. Exports the model to `model.onnx` (ONNX opset 13)
4. Saves MbTorch inference output to `mbtorch_output.json`

## Run

```bash
# From the project root:
moon run examples/export_mlp
```

Expected output:

```
Step 0: loss = 137.26980133587892
Step 49: loss = 0.621233993405312
MbTorch output:
  [0] = 5.419219650793421
  [1] = 6.9059686652151395
  [2] = 7.362450035087254
Exported ONNX to examples/export_mlp/model.onnx (355 bytes)
Saved MbTorch output to examples/export_mlp/mbtorch_output.json
```

## Verify with onnxruntime

```bash
pip install onnxruntime numpy
python3 examples/export_mlp/check_onnx_vs_mbt.py
```

This script:
1. Runs `moon run examples/export_mlp` to generate the ONNX file
2. Loads `model.onnx` with onnxruntime and runs inference
3. Compares MbTorch output vs onnxruntime output

Expected result:

```
MbTorch output:       [5.4192195 6.9059687 7.36245  ]
ONNX Runtime output:  [5.4192195 6.905969  7.36245  ]
Max abs diff:         4.77e-07

Parity check PASSED!
```

## Visualize with Netron

1. Open https://netron.app/ in your browser
2. Drag and drop `examples/export_mlp/model.onnx` onto the page
3. You will see a graph with 3 nodes:

```
input
  |
 Gemm (layer_0, transB=1)
  |
 Relu
  |
 Gemm (layer_2, transB=1)
  |
output
```

**Screenshot tips:**
- The full graph fits in one screen — capture the entire vertical layout
- Click on a Gemm node to see weight/bias shapes (e.g., weight: [4, 2], bias: [4])
- The `transB=1` attribute confirms weights are stored transposed (ONNX convention)
