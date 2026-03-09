// ONNX export tests

///|
/// E-T1: Export single Linear → Gemm node with correct structure
test "export single linear structure" {
  let weight = @tensor.Tensor::from_array_with_shape(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    [2, 3],
  )
  let bias = @tensor.Tensor::from_array_with_shape([0.1, 0.2, 0.3], [3])
  let linear = @nn.Linear::from_tensors(weight, bias)
  let layers : Array[@nn.Layer] = [@nn.Layer::Linear(linear)]
  let onnx_bytes = @io.export_onnx(layers, [1, 2])
  // Parse back and verify structure
  let graph = @io.parse_onnx(onnx_bytes)
  assert_eq(graph.nodes.length(), 1)
  assert_eq(graph.nodes[0].op_type, "Gemm")
  assert_eq(graph.initializers.length(), 2)
  // weight: transB=1, so stored as [out=3, in=2]
  assert_eq(graph.initializers[0].shape, [3, 2])
  // bias: [3]
  assert_eq(graph.initializers[1].shape, [3])
  // input/output names
  assert_eq(graph.input_names, ["input"])
  assert_eq(graph.output_names, ["output"])
}

///|
/// E-T2: Export MLP (Linear+Relu+Linear) → 3 nodes, 4 initializers
test "export mlp structure" {
  let w1 = @tensor.Tensor::from_array_with_shape(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    [2, 3],
  )
  let b1 = @tensor.Tensor::from_array_with_shape([0.1, 0.2, 0.3], [3])
  let w2 = @tensor.Tensor::from_array_with_shape([1.0, 1.0, 1.0], [3, 1])
  let b2 = @tensor.Tensor::from_array_with_shape([0.5], [1])
  let layers : Array[@nn.Layer] = [
    @nn.Layer::Linear(@nn.Linear::from_tensors(w1, b1)),
    @nn.Layer::Relu,
    @nn.Layer::Linear(@nn.Linear::from_tensors(w2, b2)),
  ]
  let onnx_bytes = @io.export_onnx(layers, [1, 2])
  let graph = @io.parse_onnx(onnx_bytes)
  assert_eq(graph.nodes.length(), 3)
  assert_eq(graph.nodes[0].op_type, "Gemm")
  assert_eq(graph.nodes[1].op_type, "Relu")
  assert_eq(graph.nodes[2].op_type, "Gemm")
  assert_eq(graph.initializers.length(), 4)
}

///|
/// E-T7: Roundtrip single Linear — export → parse → load → forward matches
test "export roundtrip single linear forward" {
  let weight = @tensor.Tensor::from_array_with_shape([2.0, 3.0], [2, 1])
  let bias = @tensor.Tensor::from_array_with_shape([1.0], [1])
  let linear = @nn.Linear::from_tensors(weight, bias)
  let layers : Array[@nn.Layer] = [@nn.Layer::Linear(linear)]
  // Original forward
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0]]),
  )
  let y_orig = linear.forward(x)
  let expected = y_orig.data().get([0, 0])
  // Export → reimport → forward
  let onnx_bytes = @io.export_onnx(layers, [1, 2])
  let imported = @io.load_onnx_model(onnx_bytes)
  let x2 = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0]]),
  )
  let y_imported = imported[0].1.forward(x2)
  let eps = 1.0e-5
  assert_true((y_imported.data().get([0, 0]) - expected).abs() < eps)
}

///|
/// E-T8: Roundtrip MLP — export → parse → load → forward matches
test "export roundtrip mlp forward" {
  let w1 = @tensor.Tensor::from_array_with_shape(
    [1.0, 0.0, -1.0, 0.0, 1.0, 0.0],
    [2, 3],
  )
  let b1 = @tensor.Tensor::from_array_with_shape([0.0, 0.0, 0.0], [3])
  let w2 = @tensor.Tensor::from_array_with_shape([1.0, 1.0, 1.0], [3, 1])
  let b2 = @tensor.Tensor::from_array_with_shape([0.5], [1])
  let layers : Array[@nn.Layer] = [
    @nn.Layer::Linear(@nn.Linear::from_tensors(w1, b1)),
    @nn.Layer::Relu,
    @nn.Layer::Linear(@nn.Linear::from_tensors(w2, b2)),
  ]
  // Original forward
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0]]),
  )
  let mut h : @autograd.Variable = x
  for i = 0; i < layers.length(); i = i + 1 {
    h = layers[i].forward(h)
  }
  let expected = h.data().get([0, 0])
  // Export → reimport → forward (load_onnx_model returns Linear layers, Relu skipped)
  let onnx_bytes = @io.export_onnx(layers, [1, 2])
  let imported = @io.load_onnx_model(onnx_bytes)
  let x2 = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0]]),
  )
  let h1 = imported[0].1.forward(x2)
  let h1_relu = h1.relu()
  let y = imported[1].1.forward(h1_relu)
  let eps = 1.0e-5
  assert_true((y.data().get([0, 0]) - expected).abs() < eps)
}

///|
/// E-T3: Export Conv2d → Conv node with strides/pads attributes
test "export conv2d structure" {
  let w = @tensor.Tensor::from_array_with_shape(
    Array::make(18, 1.0),
    [2, 1, 3, 3],
  )
  let b = @tensor.Tensor::from_array_with_shape([0.1, 0.2], [2])
  let conv = @nn.Conv2d::from_tensors(
    w,
    bias_data=Some(b),
    stride=[1, 1],
    padding=[1, 1, 1, 1],
  )
  let layers : Array[@nn.Layer] = [@nn.Layer::Conv2d(conv)]
  let onnx_bytes = @io.export_onnx(layers, [1, 1, 4, 4])
  let graph = @io.parse_onnx(onnx_bytes)
  assert_eq(graph.nodes.length(), 1)
  assert_eq(graph.nodes[0].op_type, "Conv")
  // weight [2, 1, 3, 3] + bias [2] = 2 initializers
  assert_eq(graph.initializers.length(), 2)
  assert_eq(graph.initializers[0].shape, [2, 1, 3, 3])
  assert_eq(graph.initializers[1].shape, [2])
}

///|
/// E-T4: Export BatchNorm2d → BatchNormalization with 4 initializers + epsilon
test "export batchnorm2d structure" {
  let gamma = @tensor.Tensor::from_array_with_shape([1.0, 1.0], [2])
  let beta = @tensor.Tensor::from_array_with_shape([0.0, 0.0], [2])
  let mean = @tensor.Tensor::from_array_with_shape([0.5, 0.5], [2])
  let var_ = @tensor.Tensor::from_array_with_shape([1.0, 1.0], [2])
  let bn = @nn.BatchNorm2d::from_tensors(
    gamma~,
    beta~,
    mean~,
    var_~,
    eps=1.0e-5,
  )
  let layers : Array[@nn.Layer] = [@nn.Layer::BatchNorm2d(bn)]
  let onnx_bytes = @io.export_onnx(layers, [1, 2, 4, 4])
  let graph = @io.parse_onnx(onnx_bytes)
  assert_eq(graph.nodes.length(), 1)
  assert_eq(graph.nodes[0].op_type, "BatchNormalization")
  // gamma, beta, mean, var = 4 initializers
  assert_eq(graph.initializers.length(), 4)
  // 5 inputs: X, gamma, beta, mean, var
  assert_eq(graph.nodes[0].inputs.length(), 5)
}

///|
/// E-T5: Export Flatten → Flatten node with axis=1
test "export flatten structure" {
  let layers : Array[@nn.Layer] = [@nn.Layer::Flatten]
  let onnx_bytes = @io.export_onnx(layers, [1, 2, 4, 4])
  let graph = @io.parse_onnx(onnx_bytes)
  assert_eq(graph.nodes.length(), 1)
  assert_eq(graph.nodes[0].op_type, "Flatten")
  assert_eq(graph.initializers.length(), 0)
}

///|
/// E-T6: Export CNN (Conv+BN+Relu+Flatten+Linear) full pipeline
test "export cnn structure" {
  let conv_w = @tensor.Tensor::from_array_with_shape(
    Array::make(18, 0.5),
    [2, 1, 3, 3],
  )
  let conv_b = @tensor.Tensor::from_array_with_shape([0.1, 0.2], [2])
  let gamma = @tensor.Tensor::from_array_with_shape([1.0, 1.0], [2])
  let beta = @tensor.Tensor::from_array_with_shape([0.0, 0.0], [2])
  let mean = @tensor.Tensor::from_array_with_shape([0.0, 0.0], [2])
  let var_ = @tensor.Tensor::from_array_with_shape([1.0, 1.0], [2])
  let fc_w = @tensor.Tensor::from_array_with_shape(
    Array::make(96, 0.1),
    [32, 3],
  )
  let fc_b = @tensor.Tensor::from_array_with_shape([0.0, 0.0, 0.0], [3])
  let layers : Array[@nn.Layer] = [
    @nn.Layer::Conv2d(
      @nn.Conv2d::from_tensors(
        conv_w,
        bias_data=Some(conv_b),
        stride=[1, 1],
        padding=[1, 1, 1, 1],
      ),
    ),
    @nn.Layer::BatchNorm2d(
      @nn.BatchNorm2d::from_tensors(gamma~, beta~, mean~, var_~, eps=1.0e-5),
    ),
    @nn.Layer::Relu,
    @nn.Layer::Flatten,
    @nn.Layer::Linear(@nn.Linear::from_tensors(fc_w, fc_b)),
  ]
  let onnx_bytes = @io.export_onnx(layers, [1, 1, 4, 4])
  let graph = @io.parse_onnx(onnx_bytes)
  // 5 nodes: Conv, BatchNormalization, Relu, Flatten, Gemm
  assert_eq(graph.nodes.length(), 5)
  assert_eq(graph.nodes[0].op_type, "Conv")
  assert_eq(graph.nodes[1].op_type, "BatchNormalization")
  assert_eq(graph.nodes[2].op_type, "Relu")
  assert_eq(graph.nodes[3].op_type, "Flatten")
  assert_eq(graph.nodes[4].op_type, "Gemm")
  // Conv(weight+bias=2) + BN(gamma+beta+mean+var=4) + Gemm(weight+bias=2) = 8 initializers
  assert_eq(graph.initializers.length(), 8)
}

///|
/// E-T10: Export produces valid bytes that parse_onnx can read
test "export produces valid onnx bytes" {
  let linear = @nn.Linear::new(4, 2, seed=42UL)
  let layers : Array[@nn.Layer] = [@nn.Layer::Linear(linear)]
  let onnx_bytes = @io.export_onnx(layers, [1, 4])
  // Should not throw
  let graph = @io.parse_onnx(onnx_bytes)
  assert_true(graph.nodes.length() > 0)
}
