///|
/// Train a 2-layer MLP, serialize to both JSON and Binary .mbt,
/// deserialize both, and verify outputs match.
fn main {
  // Model: Linear(2,4) -> relu -> Linear(4,1)
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)

  // Synthetic data: y = 2*x1 + 3*x2
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
  )
  let target = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[5.0], [7.0], [8.0]]),
  )

  // Train for 50 steps
  let params = l1.parameters()
  params.append(l2.parameters())
  let optimizer = @optim.SGD::new(params, lr=0.001)
  let mut initial_loss = 0.0
  let mut final_loss = 0.0
  for step = 0; step < 50; step = step + 1 {
    let pred = l2.forward(l1.forward(x).relu())
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()
    if step == 0 {
      initial_loss = loss.data().get([])
    }
    if step == 49 {
      final_loss = loss.data().get([])
    }
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  }
  println("Training: loss \{initial_loss} -> \{final_loss}")

  let layers = [("hidden", l1), ("output", l2)]

  // --- JSON .mbt ---
  let json = @io.serialize_model(layers)
  println("JSON .mbt size: \{json.length()} chars")
  let json_layers = @io.deserialize_model(json) catch {
    _ => abort("JSON deserialize failed")
  }

  // --- Binary .mbt ---
  let binary = @io.serialize_model_to_binary(layers)
  println("Binary .mbt size: \{binary.length()} bytes")
  let bin_layers = @io.deserialize_model_from_binary(binary) catch {
    _ => abort("Binary deserialize failed")
  }

  // --- Compare 3 patterns ---
  let orig = l2.forward(l1.forward(x).relu())
  let from_json = json_layers[1].1.forward(
    json_layers[0].1.forward(x).relu(),
  )
  let from_bin = bin_layers[1].1.forward(
    bin_layers[0].1.forward(x).relu(),
  )

  let mut max_diff_json = 0.0
  let mut max_diff_bin = 0.0
  for r = 0; r < 3; r = r + 1 {
    let o = orig.data().get([r, 0])
    let j = from_json.data().get([r, 0])
    let b = from_bin.data().get([r, 0])
    let dj = (o - j).abs()
    let db = (o - b).abs()
    if dj > max_diff_json {
      max_diff_json = dj
    }
    if db > max_diff_bin {
      max_diff_bin = db
    }
  }
  println("Original vs JSON:   max diff = \{max_diff_json}")
  println("Original vs Binary: max diff = \{max_diff_bin}")
  println("Roundtrip OK!")
}
