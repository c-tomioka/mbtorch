// ONNX export attribute verification tests
//
// These tests verify that exported ONNX attributes (names, types, values)
// survive a roundtrip through export_onnx → parse_onnx. They complement
// the structure-level tests in export_test.mbt by checking actual attribute
// values, not just node counts.
//
// Background: A field number bug (field 7 instead of 8 for ints attributes)
// was not caught by structure tests because they only checked node/initializer
// counts. These tests would have caught that bug.

///|
/// P0-1: Conv2d export → parse preserves strides, pads, dilations, group attributes
test "export conv2d attributes roundtrip" {
  let w = @tensor.Tensor::from_array_with_shape(
    Array::make(18, 1.0),
    [2, 1, 3, 3],
  )
  let b = @tensor.Tensor::from_array_with_shape([0.1, 0.2], [2])
  let conv = @nn.Conv2d::from_tensors(
    w,
    bias_data=Some(b),
    stride=[2, 2],
    padding=[1, 1, 1, 1],
  )
  let layers : Array[@nn.Layer] = [@nn.Layer::Conv2d(conv)]
  let onnx_bytes = @io.export_onnx(layers, [1, 1, 8, 8])
  let graph = @io.parse_onnx(onnx_bytes)
  assert_eq(graph.nodes[0].op_type, "Conv")
  let attrs = graph.nodes[0].attributes
  let mut found = 0
  for i = 0; i < attrs.length(); i = i + 1 {
    match attrs[i].name {
      "strides" => {
        assert_eq(attrs[i].ints, [2, 2])
        found = found + 1
      }
      "pads" => {
        assert_eq(attrs[i].ints, [1, 1, 1, 1])
        found = found + 1
      }
      "dilations" => {
        assert_eq(attrs[i].ints, [1, 1])
        found = found + 1
      }
      "group" => {
        assert_eq(attrs[i].int_value, 1L)
        found = found + 1
      }
      _ => ()
    }
  }
  assert_eq(found, 4)
}

///|
/// P0-1b: Conv2d with default stride=[1,1] and no padding
test "export conv2d default stride attributes roundtrip" {
  let w = @tensor.Tensor::from_array_with_shape(
    Array::make(18, 0.5),
    [2, 1, 3, 3],
  )
  let conv = @nn.Conv2d::from_tensors(w)
  let layers : Array[@nn.Layer] = [@nn.Layer::Conv2d(conv)]
  let onnx_bytes = @io.export_onnx(layers, [1, 1, 8, 8])
  let graph = @io.parse_onnx(onnx_bytes)
  assert_eq(graph.nodes[0].op_type, "Conv")
  let attrs = graph.nodes[0].attributes
  let mut found_strides = false
  let mut found_pads = false
  for i = 0; i < attrs.length(); i = i + 1 {
    match attrs[i].name {
      "strides" => {
        assert_eq(attrs[i].ints, [1, 1])
        found_strides = true
      }
      "pads" => {
        assert_eq(attrs[i].ints, [0, 0, 0, 0])
        found_pads = true
      }
      _ => ()
    }
  }
  assert_true(found_strides)
  assert_true(found_pads)
}

///|
/// P0-2: Flatten export → parse preserves axis=1 attribute
test "export flatten axis attribute roundtrip" {
  let layers : Array[@nn.Layer] = [@nn.Layer::Flatten]
  let onnx_bytes = @io.export_onnx(layers, [1, 2, 4, 4])
  let graph = @io.parse_onnx(onnx_bytes)
  assert_eq(graph.nodes[0].op_type, "Flatten")
  let attrs = graph.nodes[0].attributes
  let mut found_axis = false
  for i = 0; i < attrs.length(); i = i + 1 {
    if attrs[i].name == "axis" {
      assert_eq(attrs[i].int_value, 1L)
      found_axis = true
    }
  }
  assert_true(found_axis)
}

///|
/// P0-3: BatchNorm2d export → parse preserves epsilon float attribute
test "export batchnorm2d epsilon attribute roundtrip" {
  let bn = @nn.BatchNorm2d::from_tensors(
    gamma=@tensor.Tensor::ones([2]),
    beta=@tensor.Tensor::zeros([2]),
    mean=@tensor.Tensor::zeros([2]),
    var_=@tensor.Tensor::ones([2]),
    eps=1.0e-5,
  )
  let layers : Array[@nn.Layer] = [@nn.Layer::BatchNorm2d(bn)]
  let onnx_bytes = @io.export_onnx(layers, [1, 2, 4, 4])
  let graph = @io.parse_onnx(onnx_bytes)
  assert_eq(graph.nodes[0].op_type, "BatchNormalization")
  let attrs = graph.nodes[0].attributes
  let mut found_eps = false
  for i = 0; i < attrs.length(); i = i + 1 {
    if attrs[i].name == "epsilon" {
      assert_true((attrs[i].float_value - 1.0e-5).abs() < 1.0e-7)
      found_eps = true
    }
  }
  assert_true(found_eps)
}

///|
/// P0-4: Gemm export → parse preserves transB=1 attribute
test "export gemm transB attribute roundtrip" {
  let w = @tensor.Tensor::from_array_with_shape([1.0, 2.0, 3.0, 4.0], [2, 2])
  let b = @tensor.Tensor::from_array_with_shape([0.1, 0.2], [2])
  let linear = @nn.Linear::from_tensors(w, b)
  let layers : Array[@nn.Layer] = [@nn.Layer::Linear(linear)]
  let onnx_bytes = @io.export_onnx(layers, [1, 2])
  let graph = @io.parse_onnx(onnx_bytes)
  assert_eq(graph.nodes[0].op_type, "Gemm")
  let attrs = graph.nodes[0].attributes
  let mut found_transB = false
  for i = 0; i < attrs.length(); i = i + 1 {
    if attrs[i].name == "transB" {
      assert_eq(attrs[i].int_value, 1L)
      found_transB = true
    }
  }
  assert_true(found_transB)
}

///|
/// P0-5: SelfAttention Transpose node has perm=[1,0] attribute
test "export self_attention transpose perm attribute roundtrip" {
  let sa = @nn.SelfAttention::new(4, num_heads=1, seed=42UL)
  let layers : Array[@nn.Layer] = [@nn.Layer::SelfAttention(sa)]
  let onnx_bytes = @io.export_onnx(layers, [1, 2, 4])
  let graph = @io.parse_onnx(onnx_bytes)
  // Node 3 is Transpose (K^T)
  assert_eq(graph.nodes[3].op_type, "Transpose")
  let attrs = graph.nodes[3].attributes
  let mut found_perm = false
  for i = 0; i < attrs.length(); i = i + 1 {
    if attrs[i].name == "perm" {
      assert_eq(attrs[i].ints, [1, 0])
      found_perm = true
    }
  }
  assert_true(found_perm)
}

///|
/// P0-6: Full CNN — all nodes have correct attributes after roundtrip
test "export cnn all attributes roundtrip" {
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
  let onnx_bytes = @io.export_onnx(layers, [1, 1, 4, 4])
  let graph = @io.parse_onnx(onnx_bytes)
  // Conv node (index 0): strides, pads, dilations, group
  assert_eq(graph.nodes[0].op_type, "Conv")
  let conv_attrs = graph.nodes[0].attributes
  let mut conv_checked = 0
  for i = 0; i < conv_attrs.length(); i = i + 1 {
    match conv_attrs[i].name {
      "strides" => {
        assert_eq(conv_attrs[i].ints, [1, 1])
        conv_checked = conv_checked + 1
      }
      "pads" => {
        assert_eq(conv_attrs[i].ints, [1, 1, 1, 1])
        conv_checked = conv_checked + 1
      }
      "dilations" => {
        assert_eq(conv_attrs[i].ints, [1, 1])
        conv_checked = conv_checked + 1
      }
      "group" => {
        assert_eq(conv_attrs[i].int_value, 1L)
        conv_checked = conv_checked + 1
      }
      _ => ()
    }
  }
  assert_eq(conv_checked, 4)
  // BatchNormalization node (index 1): epsilon
  assert_eq(graph.nodes[1].op_type, "BatchNormalization")
  let bn_attrs = graph.nodes[1].attributes
  let mut found_eps = false
  for i = 0; i < bn_attrs.length(); i = i + 1 {
    if bn_attrs[i].name == "epsilon" {
      assert_true((bn_attrs[i].float_value - 1.0e-5).abs() < 1.0e-7)
      found_eps = true
    }
  }
  assert_true(found_eps)
  // Relu node (index 2): no attributes
  assert_eq(graph.nodes[2].op_type, "Relu")
  assert_eq(graph.nodes[2].attributes.length(), 0)
  // Flatten node (index 3): axis=1
  assert_eq(graph.nodes[3].op_type, "Flatten")
  let flatten_attrs = graph.nodes[3].attributes
  let mut found_axis = false
  for i = 0; i < flatten_attrs.length(); i = i + 1 {
    if flatten_attrs[i].name == "axis" {
      assert_eq(flatten_attrs[i].int_value, 1L)
      found_axis = true
    }
  }
  assert_true(found_axis)
  // Gemm node (index 4): transB=1
  assert_eq(graph.nodes[4].op_type, "Gemm")
  let gemm_attrs = graph.nodes[4].attributes
  let mut found_transB = false
  for i = 0; i < gemm_attrs.length(); i = i + 1 {
    if gemm_attrs[i].name == "transB" {
      assert_eq(gemm_attrs[i].int_value, 1L)
      found_transB = true
    }
  }
  assert_true(found_transB)
}
