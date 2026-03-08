///|
/// Train a 2-layer MLP, serialize it to .mbt JSON, deserialize, and verify.
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

  // Serialize
  let json = @io.serialize_model([("hidden", l1), ("output", l2)])
  println("Serialized JSON length: \{json.length()} chars")

  // Deserialize
  let layers = @io.deserialize_model(json) catch {
    _ => abort("deserialize failed")
  }
  let loaded_l1 = layers[0].1
  let loaded_l2 = layers[1].1

  // Compare outputs
  let orig = l2.forward(l1.forward(x).relu())
  let restored = loaded_l2.forward(loaded_l1.forward(x).relu())
  let mut max_diff = 0.0
  for r = 0; r < 3; r = r + 1 {
    let d = orig.data().get([r, 0]) - restored.data().get([r, 0])
    let abs_d = if d < 0.0 { -d } else { d }
    if abs_d > max_diff {
      max_diff = abs_d
    }
  }
  println("Max output difference after roundtrip: \{max_diff}")
  println("Roundtrip OK!")
}
