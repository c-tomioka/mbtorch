///|
/// Build a small CNN in MbTorch, export it to ONNX, and save inference output.
///
/// Model: Conv2d(1→2, 3×3) → BatchNorm2d → ReLU → Flatten → Linear(32→3)
/// Input: [1, 1, 4, 4] (batch=1, channel=1, 4×4 image)
///
/// Outputs:
///   examples/export_cnn/model.onnx          — ONNX graph (Conv → BN → Relu → Flatten → Gemm)
///   examples/export_cnn/mbtorch_output.json  — MbTorch inference result
///   examples/export_cnn/input.json           — Input tensor values (for Python verification)
fn main {
  // 1. Define CNN layers
  // Conv2d: 1 input channel → 2 output channels, 3×3 kernel, padding=[1,1,1,1] to preserve size
  let conv_weight = @tensor.Tensor::rand([2, 1, 3, 3], seed=42UL)
  let conv_bias = @tensor.Tensor::zeros([2])
  let conv = @nn.Conv2d::from_tensors(
    conv_weight,
    bias_data=Some(conv_bias),
    stride=[1, 1],
    padding=[1, 1, 1, 1],
  )

  // BatchNorm2d: 2 channels, default params (gamma=1, beta=0, mean=0, var=1)
  let bn = @nn.BatchNorm2d::from_tensors(
    gamma=@tensor.Tensor::ones([2]),
    beta=@tensor.Tensor::zeros([2]),
    mean=@tensor.Tensor::zeros([2]),
    var_=@tensor.Tensor::ones([2]),
  )

  // Linear: after Flatten, input is 2*4*4=32 → output 3 classes
  let linear = @nn.Linear::new(32, 3, seed=123UL)

  // 2. Build layer array
  let layers : Array[@nn.Layer] = [
    @nn.Layer::Conv2d(conv),
    @nn.Layer::BatchNorm2d(bn),
    @nn.Layer::Relu,
    @nn.Layer::Flatten,
    @nn.Layer::Linear(linear),
  ]

  // 3. Create test input [1, 1, 4, 4] with fixed values
  let input_data = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
    1.5, 1.6,
  ]
  let input_tensor = @tensor.Tensor::from_array_with_shape(input_data, [1, 1, 4, 4])
  let x = @autograd.Variable::new(input_tensor)

  // 4. Forward pass
  let mut out = x
  for i = 0; i < layers.length(); i = i + 1 {
    out = layers[i].forward(out)
  }
  let out_data = out.data()
  let out_shape = out_data.shape()
  println("CNN output shape: \{out_shape}")
  println("CNN output:")
  let out_arr = out_data.to_array()
  for i = 0; i < out_arr.length(); i = i + 1 {
    println("  [\{i}] = \{out_arr[i]}")
  }

  // 5. Export to ONNX
  let onnx_bytes = @io.export_onnx(layers, [1, 1, 4, 4])
  @fs.write_bytes_to_file(
    "examples/export_cnn/model.onnx",
    onnx_bytes,
  ) catch {
    e => abort("Failed to write ONNX: \{e}")
  }
  println(
    "Exported ONNX to examples/export_cnn/model.onnx (\{onnx_bytes.length()} bytes)",
  )

  // 6. Save MbTorch output as JSON
  let mut json = "["
  for i = 0; i < out_arr.length(); i = i + 1 {
    if i > 0 {
      json = json + ","
    }
    json = json + out_arr[i].to_string()
  }
  json = json + "]"
  @fs.write_string_to_file(
    "examples/export_cnn/mbtorch_output.json",
    json,
  ) catch {
    e => abort("Failed to write output JSON: \{e}")
  }
  println("Saved MbTorch output to examples/export_cnn/mbtorch_output.json")

  // 7. Save input tensor as JSON (for Python verification)
  let mut input_json = "["
  for i = 0; i < input_data.length(); i = i + 1 {
    if i > 0 {
      input_json = input_json + ","
    }
    input_json = input_json + input_data[i].to_string()
  }
  input_json = input_json + "]"
  @fs.write_string_to_file(
    "examples/export_cnn/input.json",
    input_json,
  ) catch {
    e => abort("Failed to write input JSON: \{e}")
  }
  println("Saved input to examples/export_cnn/input.json")
}
