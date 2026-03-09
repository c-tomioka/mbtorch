///|
/// E1: Single Linear — ONNX structure (no initializers) + safetensors weights
/// W = [[2.0], [3.0]] (shape [2,1]), B = [1.0] (shape [1])
/// x = [[1.0, 1.0]] → y = x @ W + B = [[6.0]]
test "e2e onnx structure with safetensors weights single linear" {
  // Build ONNX bytes: Gemm node with no initializers
  let node = test_build_node_proto("Gemm", ["X", "W", "B"], ["Y"], [])
  let graph = test_build_graph_proto([node], [], ["X"], ["Y"])
  let onnx_data = test_build_model_proto(graph)
  // Build safetensors bytes: W [2,1] and B [1]
  let st_data = build_safetensors_with_tensors([
    ("W", [2, 1], [2.0, 3.0]),
    ("B", [1], [1.0]),
  ])
  // Load model
  let layers = @io.load_model_from_onnx_and_safetensors(onnx_data, st_data)
  assert_eq(layers.length(), 1)
  assert_eq(layers[0].0, "Y")
  // Forward pass
  let x = @autograd.Variable::new(@tensor.Tensor::from_array_2d([[1.0, 1.0]]))
  let y = layers[0].1.forward(x)
  let eps = 1.0e-5
  assert_true((y.data().get([0, 0]) - 6.0).abs() < eps)
}

///|
/// E2: Two-layer MLP — ONNX structure + safetensors weights
/// W1 = [[1,0],[0,1]] (identity), B1 = [0.5, -1.5]
/// W2 = [[1],[1]], B2 = [0]
/// x = [[1,1]] → h = [[1.5, -0.5]] → relu → [[1.5, 0.0]] → y = [[1.5]]
test "e2e onnx structure with safetensors weights two layer mlp" {
  // Build ONNX bytes: Gemm → Relu → Gemm, no initializers
  let node1 = test_build_node_proto("Gemm", ["X", "W1", "B1"], ["H"], [])
  let relu = test_build_node_proto("Relu", ["H"], ["H_relu"], [])
  let node2 = test_build_node_proto("Gemm", ["H_relu", "W2", "B2"], ["Y"], [])
  let graph = test_build_graph_proto([node1, relu, node2], [], ["X"], ["Y"])
  let onnx_data = test_build_model_proto(graph)
  // Build safetensors bytes: 4 tensors
  let st_data = build_safetensors_with_tensors([
    ("W1", [2, 2], [1.0, 0.0, 0.0, 1.0]),
    ("B1", [2], [0.5, -1.5]),
    ("W2", [2, 1], [1.0, 1.0]),
    ("B2", [1], [0.0]),
  ])
  // Load model
  let layers = @io.load_model_from_onnx_and_safetensors(onnx_data, st_data)
  assert_eq(layers.length(), 2)
  assert_eq(layers[0].0, "H")
  assert_eq(layers[1].0, "Y")
  // Forward pass: Layer 1
  let x = @autograd.Variable::new(@tensor.Tensor::from_array_2d([[1.0, 1.0]]))
  let h = layers[0].1.forward(x)
  // Apply ReLU manually
  let h_relu = h.relu()
  // Layer 2
  let y = layers[1].1.forward(h_relu)
  let eps = 1.0e-5
  assert_true((y.data().get([0, 0]) - 1.5).abs() < eps)
}
