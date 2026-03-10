///|
test "binary linear roundtrip preserves output" {
  let linear = @nn.Linear::new(3, 2, seed=42UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0]]),
  )
  let original = linear.forward(x)
  let data = @io.serialize_model_to_binary([("fc1", linear)])
  let layers = @io.deserialize_model_from_binary(data)
  assert_eq(layers.length(), 1)
  assert_eq(layers[0].0, "fc1")
  let restored = layers[0].1.forward(x)
  // float32 roundtrip has limited precision
  for j = 0; j < 2; j = j + 1 {
    let diff = (original.data().get([0, j]) - restored.data().get([0, j])).abs()
    assert_true(diff < 1.0e-6)
  }
}

///|
test "binary mlp roundtrip preserves output" {
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(@tensor.Tensor::from_array_2d([[1.0, 2.0]]))
  let original = l2.forward(l1.forward(x).relu())
  let data = @io.serialize_model_to_binary([("hidden", l1), ("output", l2)])
  let layers = @io.deserialize_model_from_binary(data)
  assert_eq(layers.length(), 2)
  let loaded_l1 = layers[0].1
  let loaded_l2 = layers[1].1
  let loaded = loaded_l2.forward(loaded_l1.forward(x).relu())
  let diff = (loaded.data().get([0, 0]) - original.data().get([0, 0])).abs()
  assert_true(diff < 1.0e-5)
}

///|
test "binary format has correct header" {
  let linear = @nn.Linear::new(2, 1, seed=1UL)
  let data = @io.serialize_model_to_binary([("fc", linear)])
  // Magic: "MBTM"
  assert_eq(data[0], b'M')
  assert_eq(data[1], b'B')
  assert_eq(data[2], b'T')
  assert_eq(data[3], b'M')
  // Version: 1 (u32 LE)
  assert_eq(data[4], b'\x01')
  assert_eq(data[5], b'\x00')
  assert_eq(data[6], b'\x00')
  assert_eq(data[7], b'\x00')
}

///|
test "binary deserialize invalid magic raises error" {
  let buf : Array[Byte] = []
  // Wrong magic
  for _i = 0; _i < 20; _i = _i + 1 {
    buf.push(b'\x00')
  }
  let data = Bytes::from_array(buf)
  let mut got_error = false
  try {
    let _ = @io.deserialize_model_from_binary(data)
  } catch {
    @io.InvalidFormat(_) => got_error = true
    _ => ()
  }
  assert_true(got_error)
}

///|
test "binary deserialize future version raises error" {
  let buf : Array[Byte] = []
  // Magic "MBTM"
  buf.push(b'M')
  buf.push(b'B')
  buf.push(b'T')
  buf.push(b'M')
  // Version 99 (u32 LE)
  buf.push(b'\x63')
  buf.push(b'\x00')
  buf.push(b'\x00')
  buf.push(b'\x00')
  // metadata_size = 0
  buf.push(b'\x00')
  buf.push(b'\x00')
  buf.push(b'\x00')
  buf.push(b'\x00')
  let data = Bytes::from_array(buf)
  let mut got_error = false
  let mut version = 0
  try {
    let _ = @io.deserialize_model_from_binary(data)
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
test "binary deserialize truncated data raises error" {
  // Only 4 bytes — too short for header
  let data = Bytes::from_array([b'\x00', b'\x00', b'\x00', b'\x00'])
  let mut got_error = false
  try {
    let _ = @io.deserialize_model_from_binary(data)
  } catch {
    @io.InvalidFormat(_) => got_error = true
    _ => ()
  }
  assert_true(got_error)
}

///|
test "binary preserves layer names" {
  let l1 = @nn.Linear::new(2, 3, seed=1UL)
  let l2 = @nn.Linear::new(3, 1, seed=2UL)
  let data = @io.serialize_model_to_binary([("encoder", l1), ("decoder", l2)])
  let layers = @io.deserialize_model_from_binary(data)
  assert_eq(layers[0].0, "encoder")
  assert_eq(layers[1].0, "decoder")
}

///|
/// D-T11: Binary f16 roundtrip — serialize with F16, deserialize, compare
test "binary f16 roundtrip" {
  let linear = @nn.Linear::new(3, 2, seed=42UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0]]),
  )
  let original = linear.forward(x)
  let data = @io.serialize_model_to_binary_with_dtype([("fc1", linear)], "F16")
  let layers = @io.deserialize_model_from_binary(data)
  assert_eq(layers.length(), 1)
  let restored = layers[0].1.forward(x)
  for j = 0; j < 2; j = j + 1 {
    let diff = (original.data().get([0, j]) - restored.data().get([0, j])).abs()
    assert_true(diff < 0.1) // f16 has lower precision than f32
  }
}

///|
/// D-T12: Binary int8 roundtrip — serialize with I8, deserialize, compare
test "binary int8 roundtrip" {
  let linear = @nn.Linear::new(3, 2, seed=42UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0]]),
  )
  let original = linear.forward(x)
  let data = @io.serialize_model_to_binary_with_dtype([("fc1", linear)], "I8")
  let layers = @io.deserialize_model_from_binary(data)
  assert_eq(layers.length(), 1)
  let restored = layers[0].1.forward(x)
  for j = 0; j < 2; j = j + 1 {
    let diff = (original.data().get([0, j]) - restored.data().get([0, j])).abs()
    assert_true(diff < 1.0) // int8 has significant quantization error
  }
}

///|
/// D-T13: Binary v1 backward compat — existing F32 binary loads with new code
test "binary v1 backward compat" {
  // Serialize with original F32 function
  let linear = @nn.Linear::new(2, 1, seed=42UL)
  let data = @io.serialize_model_to_binary([("fc", linear)])
  // Deserialize with updated function (handles optional scale/zero_point)
  let layers = @io.deserialize_model_from_binary(data)
  assert_eq(layers.length(), 1)
  assert_eq(layers[0].0, "fc")
}

///|
/// D-T14: Binary f16 size reduction — f16 binary should be smaller than f32
test "binary f16 size reduction" {
  let linear = @nn.Linear::new(100, 50, seed=42UL)
  let f32_data = @io.serialize_model_to_binary([("fc", linear)])
  let f16_data = @io.serialize_model_to_binary_with_dtype([("fc", linear)], "F16")
  // f16 should be noticeably smaller (weight data is half the size)
  assert_true(f16_data.length() < f32_data.length())
  // Rough check: tensor data dominates, so f16 should be roughly 50-70% of f32
  let ratio = f16_data.length().to_double() / f32_data.length().to_double()
  assert_true(ratio < 0.75)
}

///|
test "binary json interconvert" {
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(@tensor.Tensor::from_array_2d([[1.0, 2.0]]))
  let layers = [("hidden", l1), ("output", l2)]
  // JSON roundtrip
  let json_str = @io.serialize_model(layers)
  let json_layers = @io.deserialize_model(json_str)
  let json_out = json_layers[1].1.forward(
    json_layers[0].1.forward(x).relu(),
  )
  // Binary roundtrip
  let bin_data = @io.serialize_model_to_binary(layers)
  let bin_layers = @io.deserialize_model_from_binary(bin_data)
  let bin_out = bin_layers[1].1.forward(
    bin_layers[0].1.forward(x).relu(),
  )
  // Both should produce similar results (binary has float32 precision loss)
  let diff = (json_out.data().get([0, 0]) - bin_out.data().get([0, 0])).abs()
  assert_true(diff < 1.0e-5)
}
