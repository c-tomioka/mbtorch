# export_cnn — MbTorch CNN ONNX Export Demo

Build a small CNN in MbTorch and export it to ONNX. Then verify the exported model with onnxruntime and visualize it with Netron.

## What it does

1. Defines a CNN: `Conv2d(1->2, 3x3, pad=1) -> BatchNorm2d(2) -> ReLU -> Flatten -> Linear(32->3)`
2. Runs inference on a fixed [1, 1, 4, 4] input (no training — inference only)
3. Exports the model to `model.onnx` (ONNX opset 13)
4. Saves MbTorch inference output to `mbtorch_output.json`
5. Saves input tensor to `input.json` (for Python verification)

## Run

```bash
# From the project root:
moon run examples/export_cnn
```

Expected output:

```
CNN output shape: [1, 3]
CNN output:
  [0] = 0.7065596592220158
  [1] = -0.14316693727930596
  [2] = -0.45241792006163073
Exported ONNX to examples/export_cnn/model.onnx (1118 bytes)
Saved MbTorch output to examples/export_cnn/mbtorch_output.json
Saved input to examples/export_cnn/input.json
```

## Verify with onnxruntime

```bash
pip install onnxruntime numpy
python3 examples/export_cnn/check_onnx_vs_mbt.py
```

This script:
1. Runs `moon run examples/export_cnn` to generate the ONNX file
2. Loads `model.onnx` with onnxruntime and runs inference with the same input
3. Compares MbTorch output vs onnxruntime output

Expected result:

```
MbTorch output:       [ 0.70655966 -0.14316694 -0.4524179 ]
ONNX Runtime output:  [ 0.70655954 -0.14316688 -0.45241803]
Max abs diff:         1.19e-07

Parity check PASSED!
```

## Visualize with Netron

1. Open https://netron.app/ in your browser
2. Drag and drop `examples/export_cnn/model.onnx` onto the page
3. You will see a graph with 5 nodes:

```
input
  |
 Conv (layer_0)
  |
 BatchNormalization (layer_1)
  |
 Relu
  |
 Flatten (axis=1)
  |
 Gemm (layer_4, transB=1)
  |
output
```

**Screenshot tips:**
- The full graph fits in one screen — capture the entire vertical layout
- Click on the Conv node to see weight shape [2, 1, 3, 3], strides [1, 1], pads [1, 1, 1, 1]
- Click on BatchNormalization to see gamma/beta/mean/var shapes [2]
- The Flatten node has `axis=1`, collapsing spatial dims into a flat vector of size 32
