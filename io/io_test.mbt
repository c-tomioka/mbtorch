///|
test "linear roundtrip preserves output" {
  let linear = @nn.Linear::new(3, 2, seed=42UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0]]),
  )
  let original_output = linear.forward(x)
  let json = @io.serialize_linear(linear, name="fc1")
  let loaded = @io.deserialize_linear(json)
  let loaded_output = loaded.forward(x)
  assert_eq(
    loaded_output.data().get([0, 0]),
    original_output.data().get([0, 0]),
  )
  assert_eq(
    loaded_output.data().get([0, 1]),
    original_output.data().get([0, 1]),
  )
}

///|
test "mlp roundtrip preserves output" {
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(@tensor.Tensor::from_array_2d([[1.0, 2.0]]))
  let original = l2.forward(l1.forward(x).relu())
  let json = @io.serialize_model([("hidden", l1), ("output", l2)])
  let layers = @io.deserialize_model(json)
  let loaded_l1 = layers[0].1
  let loaded_l2 = layers[1].1
  let loaded = loaded_l2.forward(loaded_l1.forward(x).relu())
  assert_eq(loaded.data().get([0, 0]), original.data().get([0, 0]))
}

///|
test "serialize includes format and version" {
  let linear = @nn.Linear::new(2, 1)
  let json = @io.serialize_linear(linear)
  assert_true(json.contains("\"mbtorch\""))
  assert_true(json.contains("\"version\""))
}

///|
test "deserialize invalid json raises error" {
  let mut got_error = false
  try {
    let _ = @io.deserialize_linear("not json")
  } catch {
    @io.InvalidFormat(_) => got_error = true
    _ => ()
  }
  assert_true(got_error)
}

///|
test "deserialize wrong format raises error" {
  let json =
    #|{"format":"not_mbtorch","version":1,"layers":[]}
  let mut got_error = false
  try {
    let _ = @io.deserialize_linear(json)
  } catch {
    @io.InvalidFormat(_) => got_error = true
    _ => ()
  }
  assert_true(got_error)
}

///|
test "deserialize future version raises error" {
  let json =
    #|{"format":"mbtorch","version":99,"layers":[]}
  let mut got_error = false
  let mut version = 0
  try {
    let _ = @io.deserialize_model(json)
  } catch {
    @io.UnsupportedVersion(v) => {
      got_error = true
      version = v
    }
    _ => ()
  }
  assert_true(got_error)
  assert_eq(version, 99)
}

///|
test "deserialize preserves layer names" {
  let l1 = @nn.Linear::new(2, 3, seed=1UL)
  let l2 = @nn.Linear::new(3, 1, seed=2UL)
  let json = @io.serialize_model([("encoder", l1), ("decoder", l2)])
  let layers = @io.deserialize_model(json)
  assert_eq(layers[0].0, "encoder")
  assert_eq(layers[1].0, "decoder")
}

///|
test "roundtrip with trained model" {
  let linear = @nn.Linear::new(2, 1, seed=42UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0]]),
  )
  let target = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[5.0], [7.0]]),
  )
  // Train a few steps
  let optimizer = @optim.SGD::new(linear.parameters(), lr=0.01)
  for _step = 0; _step < 20; _step = _step + 1 {
    let pred = linear.forward(x)
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  }
  // Save and reload
  let json = @io.serialize_linear(linear, name="trained")
  let loaded = @io.deserialize_linear(json)
  // Compare outputs
  let orig = linear.forward(x)
  let reloaded = loaded.forward(x)
  assert_eq(reloaded.data().get([0, 0]), orig.data().get([0, 0]))
  assert_eq(reloaded.data().get([1, 0]), orig.data().get([1, 0]))
}
