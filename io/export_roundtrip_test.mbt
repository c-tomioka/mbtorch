// ONNX export roundtrip tests for CNN/BN layers
//
// These tests verify that weight data survives export→parse roundtrip
// and that forward pass results match after reconstructing layers from
// the parsed OnnxGraph. They complement the MLP roundtrip tests in
// export_test.mbt by covering Conv2d, BatchNorm2d, and full CNN models.

///|
/// Helper: find an initializer by name in the parsed graph
fn find_init(
  graph : @io.OnnxGraph,
  name : String,
) -> @io.OnnxTensor? {
  for i = 0; i < graph.initializers.length(); i = i + 1 {
    if graph.initializers[i].name == name {
      return Some(graph.initializers[i])
    }
  }
  None
}

///|
/// P1-1: Conv2d weight/bias values preserved through export → parse
test "conv2d export parse preserves weights" {
  let w_data = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
    1.5, 1.6, 1.7, 1.8,
  ]
  let b_data = [0.01, 0.02]
  let w = @tensor.Tensor::from_array_with_shape(w_data, [2, 1, 3, 3])
  let b = @tensor.Tensor::from_array_with_shape(b_data, [2])
  let conv = @nn.Conv2d::from_tensors(
    w,
    bias_data=Some(b),
    stride=[1, 1],
    padding=[1, 1, 1, 1],
  )
  let layers : Array[@nn.Layer] = [@nn.Layer::Conv2d(conv)]
  let onnx_bytes = @io.export_onnx(layers, [1, 1, 4, 4])
  let graph = @io.parse_onnx(onnx_bytes)
  // Verify weight initializer
  let w_init = find_init(graph, "layer_0_weight")
  assert_true(w_init is Some(_))
  let wi = w_init.unwrap()
  assert_eq(wi.shape, [2, 1, 3, 3])
  assert_eq(wi.data.length(), 18)
  let eps = 1.0e-5
  for i = 0; i < w_data.length(); i = i + 1 {
    assert_true((wi.data[i] - w_data[i]).abs() < eps)
  }
  // Verify bias initializer
  let b_init = find_init(graph, "layer_0_bias")
  assert_true(b_init is Some(_))
  let bi = b_init.unwrap()
  assert_eq(bi.shape, [2])
  for i = 0; i < b_data.length(); i = i + 1 {
    assert_true((bi.data[i] - b_data[i]).abs() < eps)
  }
}

///|
/// P1-2: BatchNorm2d gamma/beta/mean/var preserved through export → parse
test "batchnorm2d export parse preserves params" {
  let gamma_data = [1.5, 2.0]
  let beta_data = [0.1, -0.1]
  let mean_data = [0.5, 0.3]
  let var_data = [1.2, 0.8]
  let bn = @nn.BatchNorm2d::from_tensors(
    gamma=@tensor.Tensor::from_array_with_shape(gamma_data, [2]),
    beta=@tensor.Tensor::from_array_with_shape(beta_data, [2]),
    mean=@tensor.Tensor::from_array_with_shape(mean_data, [2]),
    var_=@tensor.Tensor::from_array_with_shape(var_data, [2]),
  )
  let layers : Array[@nn.Layer] = [@nn.Layer::BatchNorm2d(bn)]
  let onnx_bytes = @io.export_onnx(layers, [1, 2, 4, 4])
  let graph = @io.parse_onnx(onnx_bytes)
  let eps = 1.0e-5
  // gamma
  let g = find_init(graph, "layer_0_gamma").unwrap()
  assert_eq(g.shape, [2])
  for i = 0; i < gamma_data.length(); i = i + 1 {
    assert_true((g.data[i] - gamma_data[i]).abs() < eps)
  }
  // beta
  let bt = find_init(graph, "layer_0_beta").unwrap()
  for i = 0; i < beta_data.length(); i = i + 1 {
    assert_true((bt.data[i] - beta_data[i]).abs() < eps)
  }
  // running_mean
  let m = find_init(graph, "layer_0_running_mean").unwrap()
  for i = 0; i < mean_data.length(); i = i + 1 {
    assert_true((m.data[i] - mean_data[i]).abs() < eps)
  }
  // running_var
  let v = find_init(graph, "layer_0_running_var").unwrap()
  for i = 0; i < var_data.length(); i = i + 1 {
    assert_true((v.data[i] - var_data[i]).abs() < eps)
  }
}

///|
/// P1-3: Full CNN export → parse → reconstruct → forward matches original
test "cnn export parse roundtrip forward" {
  // 1. Build CNN: Conv2d → BatchNorm2d → Relu → Flatten → Linear
  let conv = @nn.Conv2d::from_tensors(
    @tensor.Tensor::rand([2, 1, 3, 3], seed=42UL),
    bias_data=Some(@tensor.Tensor::zeros([2])),
    stride=[1, 1],
    padding=[1, 1, 1, 1],
  )
  let bn = @nn.BatchNorm2d::from_tensors(
    gamma=@tensor.Tensor::ones([2]),
    beta=@tensor.Tensor::zeros([2]),
    mean=@tensor.Tensor::zeros([2]),
    var_=@tensor.Tensor::ones([2]),
  )
  let linear = @nn.Linear::new(32, 3, seed=123UL)
  let layers : Array[@nn.Layer] = [
    @nn.Layer::Conv2d(conv),
    @nn.Layer::BatchNorm2d(bn),
    @nn.Layer::Relu,
    @nn.Layer::Flatten,
    @nn.Layer::Linear(linear),
  ]
  // 2. Original forward pass
  let input = @tensor.Tensor::from_array_with_shape(
    Array::make(16, 0.5),
    [1, 1, 4, 4],
  )
  let x = @autograd.Variable::new(input)
  let mut out = x
  for i = 0; i < layers.length(); i = i + 1 {
    out = layers[i].forward(out)
  }
  let original = out.data().to_array()
  // 3. Export → parse
  let onnx_bytes = @io.export_onnx(layers, [1, 1, 4, 4])
  let graph = @io.parse_onnx(onnx_bytes)
  // 4. Reconstruct layers from parsed graph initializers
  let cw = find_init(graph, "layer_0_weight").unwrap()
  let cb = find_init(graph, "layer_0_bias").unwrap()
  let conv2 = @nn.Conv2d::from_tensors(
    @tensor.Tensor::from_array_with_shape(cw.data, cw.shape),
    bias_data=Some(@tensor.Tensor::from_array_with_shape(cb.data, cb.shape)),
    stride=[1, 1],
    padding=[1, 1, 1, 1],
  )
  let bg = find_init(graph, "layer_1_gamma").unwrap()
  let bb = find_init(graph, "layer_1_beta").unwrap()
  let bm = find_init(graph, "layer_1_running_mean").unwrap()
  let bv = find_init(graph, "layer_1_running_var").unwrap()
  let bn2 = @nn.BatchNorm2d::from_tensors(
    gamma=@tensor.Tensor::from_array_with_shape(bg.data, bg.shape),
    beta=@tensor.Tensor::from_array_with_shape(bb.data, bb.shape),
    mean=@tensor.Tensor::from_array_with_shape(bm.data, bm.shape),
    var_=@tensor.Tensor::from_array_with_shape(bv.data, bv.shape),
  )
  // Linear weight was stored transposed (transB=1), so transpose back
  let lw_init = find_init(graph, "layer_4_weight").unwrap()
  let lb_init = find_init(graph, "layer_4_bias").unwrap()
  let lw_t = @tensor.Tensor::from_array_with_shape(lw_init.data, lw_init.shape)
  let lb_t = @tensor.Tensor::from_array_with_shape(lb_init.data, lb_init.shape)
  let linear2 = @nn.Linear::from_tensors(lw_t.transpose(), lb_t)
  let layers2 : Array[@nn.Layer] = [
    @nn.Layer::Conv2d(conv2),
    @nn.Layer::BatchNorm2d(bn2),
    @nn.Layer::Relu,
    @nn.Layer::Flatten,
    @nn.Layer::Linear(linear2),
  ]
  // 5. Forward pass on reconstructed model
  let input2 = @tensor.Tensor::from_array_with_shape(
    Array::make(16, 0.5),
    [1, 1, 4, 4],
  )
  let x2 = @autograd.Variable::new(input2)
  let mut out2 = x2
  for i = 0; i < layers2.length(); i = i + 1 {
    out2 = layers2[i].forward(out2)
  }
  let reconstructed = out2.data().to_array()
  // 6. Compare
  let eps = 1.0e-5
  assert_eq(original.length(), reconstructed.length())
  for i = 0; i < original.length(); i = i + 1 {
    assert_true((original[i] - reconstructed[i]).abs() < eps)
  }
}

///|
/// P2-3: Conv2d without bias — only 1 initializer, 2 node inputs
test "conv2d no bias export roundtrip" {
  let w = @tensor.Tensor::from_array_with_shape(
    Array::make(18, 0.5),
    [2, 1, 3, 3],
  )
  let conv = @nn.Conv2d::from_tensors(w)
  let layers : Array[@nn.Layer] = [@nn.Layer::Conv2d(conv)]
  let onnx_bytes = @io.export_onnx(layers, [1, 1, 4, 4])
  let graph = @io.parse_onnx(onnx_bytes)
  assert_eq(graph.nodes[0].op_type, "Conv")
  // No bias → only weight initializer
  assert_eq(graph.initializers.length(), 1)
  assert_eq(graph.initializers[0].shape, [2, 1, 3, 3])
  // Node inputs: [input, weight] (no bias)
  assert_eq(graph.nodes[0].inputs.length(), 2)
}

///|
/// P1-5: CNN with F16 dtype — initializers have dtype=10 and values approximate
test "cnn f16 export roundtrip" {
  let conv = @nn.Conv2d::from_tensors(
    @tensor.Tensor::rand([2, 1, 3, 3], seed=42UL),
    bias_data=Some(@tensor.Tensor::zeros([2])),
    stride=[1, 1],
    padding=[1, 1, 1, 1],
  )
  let bn = @nn.BatchNorm2d::from_tensors(
    gamma=@tensor.Tensor::ones([2]),
    beta=@tensor.Tensor::zeros([2]),
    mean=@tensor.Tensor::zeros([2]),
    var_=@tensor.Tensor::ones([2]),
  )
  let linear = @nn.Linear::new(32, 3, seed=123UL)
  let layers : Array[@nn.Layer] = [
    @nn.Layer::Conv2d(conv),
    @nn.Layer::BatchNorm2d(bn),
    @nn.Layer::Relu,
    @nn.Layer::Flatten,
    @nn.Layer::Linear(linear),
  ]
  let onnx_bytes = @io.export_onnx_with_dtype(
    layers,
    [1, 1, 4, 4],
    @tensor.Float16,
  )
  let graph = @io.parse_onnx(onnx_bytes)
  // All initializers should have dtype=10 (FLOAT16)
  for i = 0; i < graph.initializers.length(); i = i + 1 {
    assert_eq(graph.initializers[i].dtype, 10)
  }
  // Verify weight values are approximately correct (F16 loses precision)
  let cw = find_init(graph, "layer_0_weight").unwrap()
  assert_eq(cw.shape, [2, 1, 3, 3])
  // F16 precision: values should be within 0.01 of original
  let original_w = @tensor.Tensor::rand([2, 1, 3, 3], seed=42UL).to_array()
  let eps = 0.01
  for i = 0; i < original_w.length(); i = i + 1 {
    assert_true((cw.data[i] - original_w[i]).abs() < eps)
  }
}
