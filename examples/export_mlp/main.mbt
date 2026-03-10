///|
/// Train a small MLP in MbTorch, export it to ONNX, and save inference output.
///
/// Outputs:
///   examples/export_mlp/model.onnx          — ONNX graph (Gemm → Relu → Gemm)
///   examples/export_mlp/mbtorch_output.json  — MbTorch inference result
fn main {
  // 1. Define model: Linear(2,4) -> ReLU -> Linear(4,1)
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)

  // 2. Train on synthetic data: y ≈ 2*x1 + 3*x2
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
  )
  let target = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[5.0], [7.0], [8.0]]),
  )
  let params = l1.parameters()
  params.append(l2.parameters())
  let optimizer = @optim.SGD::new(params, lr=0.001)
  for step = 0; step < 50; step = step + 1 {
    let pred = l2.forward(l1.forward(x).relu())
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()
    if step == 0 || step == 49 {
      println("Step \{step}: loss = \{loss.data().get([])}")
    }
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  }

  // 3. Inference with test input
  let test_input = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
  )
  let output = l2.forward(l1.forward(test_input).relu())
  let out_data = output.data()
  println("MbTorch output:")
  for r = 0; r < 3; r = r + 1 {
    println("  [\{r}] = \{out_data.get([r, 0])}")
  }

  // 4. Export to ONNX
  let layers : Array[@nn.Layer] = [
    @nn.Layer::Linear(l1),
    @nn.Layer::Relu,
    @nn.Layer::Linear(l2),
  ]
  let onnx_bytes = @io.export_onnx(layers, [1, 2])
  @fs.write_bytes_to_file(
    "examples/export_mlp/model.onnx",
    onnx_bytes,
  ) catch {
    e => abort("Failed to write ONNX: \{e}")
  }
  println("Exported ONNX to examples/export_mlp/model.onnx (\{onnx_bytes.length()} bytes)")

  // 5. Save MbTorch output as JSON for Python verification
  //    Format: [[v1],[v2],[v3]]
  let mut json = "["
  for r = 0; r < 3; r = r + 1 {
    if r > 0 {
      json = json + ","
    }
    json = json + "[\{out_data.get([r, 0])}]"
  }
  json = json + "]"
  @fs.write_string_to_file(
    "examples/export_mlp/mbtorch_output.json",
    json,
  ) catch {
    e => abort("Failed to write JSON: \{e}")
  }
  println("Saved MbTorch output to examples/export_mlp/mbtorch_output.json")
}
