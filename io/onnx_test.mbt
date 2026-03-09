// ONNX parser tests
// Note: encoding helpers are duplicated here because _test.mbt files are
// blackbox tests and cannot access package-private functions from onnx_encode.mbt.
// The canonical production versions live in onnx_encode.mbt for use by export.mbt.

///|
fn test_encode_varint(value : UInt64) -> Array[Byte] {
  let result : Array[Byte] = []
  let mut v = value
  while v >= 0x80UL {
    result.push(((v.to_int() & 0x7F) | 0x80).to_byte())
    v = v >> 7
  }
  result.push((v.to_int() & 0x7F).to_byte())
  result
}

///|
fn test_encode_tag(field_num : Int, wire_type : Int) -> Array[Byte] {
  test_encode_varint(((field_num << 3) | wire_type).to_uint64())
}

///|
fn test_append_all(target : Array[Byte], source : Array[Byte]) -> Unit {
  for i = 0; i < source.length(); i = i + 1 {
    target.push(source[i])
  }
}

///|
fn test_encode_len_delimited(
  field_num : Int,
  content : Array[Byte],
) -> Array[Byte] {
  let result : Array[Byte] = []
  test_append_all(result, test_encode_tag(field_num, 2))
  test_append_all(result, test_encode_varint(content.length().to_uint64()))
  test_append_all(result, content)
  result
}

///|
fn test_encode_varint_field(field_num : Int, value : UInt64) -> Array[Byte] {
  let result : Array[Byte] = []
  test_append_all(result, test_encode_tag(field_num, 0))
  test_append_all(result, test_encode_varint(value))
  result
}

///|
fn test_encode_string_field(field_num : Int, s : String) -> Array[Byte] {
  let bytes : Array[Byte] = []
  for i = 0; i < s.length(); i = i + 1 {
    bytes.push(s[i].to_int().to_byte())
  }
  test_encode_len_delimited(field_num, bytes)
}

///|
fn test_float32_to_le_bytes(value : Double) -> Array[Byte] {
  if value == 0.0 {
    return [b'\x00', b'\x00', b'\x00', b'\x00']
  }
  let negative = value < 0.0
  let abs_val = if negative { -value } else { value }
  let mut exp = 127
  let mut mantissa_f = abs_val
  if mantissa_f >= 2.0 {
    while mantissa_f >= 2.0 {
      mantissa_f = mantissa_f / 2.0
      exp = exp + 1
    }
  } else if mantissa_f < 1.0 {
    while mantissa_f < 1.0 && exp > 0 {
      mantissa_f = mantissa_f * 2.0
      exp = exp - 1
    }
  }
  let frac = mantissa_f - 1.0
  let mantissa_bits = (frac * 8388608.0 + 0.5).to_int()
  let sign_bit = if negative { 1 } else { 0 }
  let bits = (sign_bit << 31) | (exp << 23) | mantissa_bits
  [
    (bits & 0xFF).to_byte(),
    ((bits >> 8) & 0xFF).to_byte(),
    ((bits >> 16) & 0xFF).to_byte(),
    ((bits >> 24) & 0xFF).to_byte(),
  ]
}

///|
fn test_build_tensor_proto(
  name : String,
  shape : Array[Int],
  values : Array[Double],
) -> Array[Byte] {
  let content : Array[Byte] = []
  let dims_content : Array[Byte] = []
  for i = 0; i < shape.length(); i = i + 1 {
    test_append_all(dims_content, test_encode_varint(shape[i].to_uint64()))
  }
  test_append_all(content, test_encode_len_delimited(1, dims_content))
  test_append_all(content, test_encode_varint_field(2, 1UL))
  let float_content : Array[Byte] = []
  for i = 0; i < values.length(); i = i + 1 {
    test_append_all(float_content, test_float32_to_le_bytes(values[i]))
  }
  test_append_all(content, test_encode_len_delimited(4, float_content))
  test_append_all(content, test_encode_string_field(8, name))
  content
}

///|
fn test_build_node_proto(
  op_type : String,
  inputs : Array[String],
  outputs : Array[String],
  int_attrs : Array[(String, Int64)],
) -> Array[Byte] {
  let content : Array[Byte] = []
  for i = 0; i < inputs.length(); i = i + 1 {
    test_append_all(content, test_encode_string_field(1, inputs[i]))
  }
  for i = 0; i < outputs.length(); i = i + 1 {
    test_append_all(content, test_encode_string_field(2, outputs[i]))
  }
  test_append_all(content, test_encode_string_field(4, op_type))
  for i = 0; i < int_attrs.length(); i = i + 1 {
    let (attr_name, attr_val) = int_attrs[i]
    let attr_content : Array[Byte] = []
    test_append_all(attr_content, test_encode_string_field(1, attr_name))
    test_append_all(
      attr_content,
      test_encode_varint_field(3, attr_val.to_int().to_uint64()),
    )
    test_append_all(content, test_encode_len_delimited(5, attr_content))
  }
  content
}

///|
fn test_build_value_info(name : String) -> Array[Byte] {
  test_encode_string_field(1, name)
}

///|
fn test_build_graph_proto(
  nodes : Array[Array[Byte]],
  inits : Array[Array[Byte]],
  inputs : Array[String],
  outputs : Array[String],
) -> Array[Byte] {
  let content : Array[Byte] = []
  for i = 0; i < nodes.length(); i = i + 1 {
    test_append_all(content, test_encode_len_delimited(1, nodes[i]))
  }
  for i = 0; i < inits.length(); i = i + 1 {
    test_append_all(content, test_encode_len_delimited(5, inits[i]))
  }
  for i = 0; i < inputs.length(); i = i + 1 {
    test_append_all(
      content,
      test_encode_len_delimited(11, test_build_value_info(inputs[i])),
    )
  }
  for i = 0; i < outputs.length(); i = i + 1 {
    test_append_all(
      content,
      test_encode_len_delimited(12, test_build_value_info(outputs[i])),
    )
  }
  content
}

///|
fn test_build_model_proto(graph : Array[Byte]) -> Bytes {
  let content : Array[Byte] = []
  test_append_all(content, test_encode_len_delimited(7, graph))
  Bytes::from_array(content)
}

// === Tests ===

///|
/// T1: Parse a single Gemm node graph
test "parse_onnx single gemm node" {
  let weight = test_build_tensor_proto(
    "W",
    [2, 3],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
  )
  let bias = test_build_tensor_proto("B", [3], [0.1, 0.2, 0.3])
  let node = test_build_node_proto(
    "Gemm",
    ["X", "W", "B"],
    ["Y"],
    [("transB", 1L)],
  )
  let graph = test_build_graph_proto([node], [weight, bias], ["X"], ["Y"])
  let model_bytes = test_build_model_proto(graph)
  let g = @io.parse_onnx(model_bytes)
  assert_eq(g.nodes.length(), 1)
  assert_eq(g.nodes[0].op_type, "Gemm")
  assert_eq(g.nodes[0].inputs, ["X", "W", "B"])
  assert_eq(g.nodes[0].outputs, ["Y"])
  assert_eq(g.initializers.length(), 2)
  assert_eq(g.initializers[0].name, "W")
  assert_eq(g.initializers[0].shape, [2, 3])
  assert_eq(g.initializers[1].name, "B")
}

///|
/// T2: Load single Gemm → Linear, verify weight/bias values
test "load_onnx_model single linear" {
  let weight = test_build_tensor_proto(
    "W",
    [3, 2],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
  )
  let bias = test_build_tensor_proto("B", [2], [0.1, 0.2])
  let node = test_build_node_proto("Gemm", ["X", "W", "B"], ["Y"], [])
  let graph = test_build_graph_proto([node], [weight, bias], ["X"], ["Y"])
  let model_bytes = test_build_model_proto(graph)
  let layers = @io.load_onnx_model(model_bytes)
  assert_eq(layers.length(), 1)
  let (name, linear) = layers[0]
  assert_eq(name, "Y")
  let params = linear.parameters()
  assert_eq(params[0].shape(), [3, 2])
  assert_eq(params[1].shape(), [2])
}

///|
/// T3: Load Gemm with transB=1, verify weight is transposed
test "load_onnx_model gemm transB" {
  let weight = test_build_tensor_proto(
    "W",
    [2, 3],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
  )
  let bias = test_build_tensor_proto("B", [2], [0.1, 0.2])
  let node = test_build_node_proto(
    "Gemm",
    ["X", "W", "B"],
    ["Y"],
    [("transB", 1L)],
  )
  let graph = test_build_graph_proto([node], [weight, bias], ["X"], ["Y"])
  let model_bytes = test_build_model_proto(graph)
  let layers = @io.load_onnx_model(model_bytes)
  assert_eq(layers.length(), 1)
  let params = layers[0].1.parameters()
  assert_eq(params[0].shape(), [3, 2])
  let eps = 1.0e-6
  let w = params[0].data().to_array()
  assert_true((w[0] - 1.0).abs() < eps)
  assert_true((w[1] - 4.0).abs() < eps)
  assert_true((w[2] - 2.0).abs() < eps)
  assert_true((w[3] - 5.0).abs() < eps)
}

///|
/// T4: Two Gemm + Relu → 2 Linear layers
test "load_onnx_model two layer mlp" {
  let w1 = test_build_tensor_proto("W1", [2, 4], [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
  ])
  let b1 = test_build_tensor_proto("B1", [4], [0.1, 0.2, 0.3, 0.4])
  let w2 = test_build_tensor_proto("W2", [4, 1], [1.0, 1.0, 1.0, 1.0])
  let b2 = test_build_tensor_proto("B2", [1], [0.0])
  let node1 = test_build_node_proto("Gemm", ["X", "W1", "B1"], ["H"], [])
  let relu = test_build_node_proto("Relu", ["H"], ["H_relu"], [])
  let node2 = test_build_node_proto("Gemm", ["H_relu", "W2", "B2"], ["Y"], [])
  let graph = test_build_graph_proto(
    [node1, relu, node2],
    [w1, b1, w2, b2],
    ["X"],
    ["Y"],
  )
  let model_bytes = test_build_model_proto(graph)
  let layers = @io.load_onnx_model(model_bytes)
  assert_eq(layers.length(), 2)
  assert_eq(layers[0].0, "H")
  assert_eq(layers[1].0, "Y")
}

///|
/// T5: Reject unsupported op
test "load_onnx_model rejects unsupported op" {
  let node = test_build_node_proto("Conv", ["X"], ["Y"], [])
  let graph = test_build_graph_proto([node], [], ["X"], ["Y"])
  let model_bytes = test_build_model_proto(graph)
  let mut got_error = false
  try {
    let _ = @io.load_onnx_model(model_bytes)
  } catch {
    @io.UnsupportedOp(_) => got_error = true
    _ => ()
  }
  assert_true(got_error)
}

///|
/// T6: Reject non-float dtype
test "load_onnx_model rejects non_float" {
  let content : Array[Byte] = []
  let dims_content : Array[Byte] = []
  test_append_all(dims_content, test_encode_varint(2UL))
  test_append_all(content, test_encode_len_delimited(1, dims_content))
  test_append_all(content, test_encode_varint_field(2, 7UL))
  test_append_all(content, test_encode_string_field(8, "bad_tensor"))
  let node = test_build_node_proto(
    "Gemm",
    ["X", "bad_tensor", "B"],
    ["Y"],
    [],
  )
  let bias = test_build_tensor_proto("B", [2], [0.0, 0.0])
  let graph = test_build_graph_proto([node], [content, bias], ["X"], ["Y"])
  let model_bytes = test_build_model_proto(graph)
  let mut got_error = false
  try {
    let _ = @io.load_onnx_model(model_bytes)
  } catch {
    @io.UnsupportedDtype(_) => got_error = true
    _ => ()
  }
  assert_true(got_error)
}

///|
/// T7: Forward pass with loaded model matches manual computation
test "load_onnx_model forward matches" {
  let weight = test_build_tensor_proto("W", [2, 1], [2.0, 3.0])
  let bias = test_build_tensor_proto("B", [1], [1.0])
  let node = test_build_node_proto("Gemm", ["X", "W", "B"], ["Y"], [])
  let graph = test_build_graph_proto([node], [weight, bias], ["X"], ["Y"])
  let model_bytes = test_build_model_proto(graph)
  let layers = @io.load_onnx_model(model_bytes)
  let linear = layers[0].1
  let x = @autograd.Variable::new(@tensor.Tensor::from_array_2d([[1.0, 1.0]]))
  let y = linear.forward(x)
  let eps = 1.0e-5
  assert_true((y.data().get([0, 0]) - 6.0).abs() < eps)
}
