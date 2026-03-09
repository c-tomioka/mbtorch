///|
test "conv2d forward output shape" {
  let weight = @tensor.Tensor::from_array_with_shape(
    Array::make(2 * 3 * 3 * 3, 0.1), [2, 3, 3, 3],
  )
  let conv = @nn.Conv2d::from_tensors(weight, stride=[1, 1], padding=[0, 0, 0, 0])
  let input = @autograd.Variable::new(
    @tensor.Tensor::from_array_with_shape(Array::make(1 * 3 * 5 * 5, 1.0), [1, 3, 5, 5]),
  )
  let output = conv.forward(input)
  // out_H = (5 - 3) / 1 + 1 = 3, out_W = 3
  assert_eq(output.shape(), [1, 2, 3, 3])
}

///|
test "conv2d from_tensors known weights" {
  // Input: N=1, C=1, H=2, W=2, values=[1,2,3,4]
  // Kernel: C_out=1, C_in=1, kH=1, kW=1, weight=2.0
  let weight = @tensor.Tensor::from_array_with_shape([2.0], [1, 1, 1, 1])
  let bias = @tensor.Tensor::from_array([10.0])
  let conv = @nn.Conv2d::from_tensors(weight, bias_data=Some(bias))
  let input = @autograd.Variable::new(
    @tensor.Tensor::from_array_with_shape([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
  )
  let output = conv.forward(input)
  assert_eq(output.shape(), [1, 1, 2, 2])
  // 2*1 + 10 = 12, 2*2 + 10 = 14, 2*3 + 10 = 16, 2*4 + 10 = 18
  assert_eq(output.data().get([0, 0, 0, 0]), 12.0)
  assert_eq(output.data().get([0, 0, 0, 1]), 14.0)
  assert_eq(output.data().get([0, 0, 1, 0]), 16.0)
  assert_eq(output.data().get([0, 0, 1, 1]), 18.0)
}

///|
test "batchnorm2d identity" {
  // gamma=1, beta=0, mean=0, var=1, eps=0 => output = input
  let input = @autograd.Variable::new(
    @tensor.Tensor::from_array_with_shape([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
  )
  let bn = @nn.BatchNorm2d::from_tensors(
    gamma=@tensor.Tensor::from_array([1.0]),
    beta=@tensor.Tensor::from_array([0.0]),
    mean=@tensor.Tensor::from_array([0.0]),
    var_=@tensor.Tensor::from_array([1.0]),
    eps=0.0,
  )
  let out = bn.forward(input)
  assert_eq(out.data().shape(), [1, 1, 2, 2])
  assert_eq(out.data().get([0, 0, 0, 0]), 1.0)
  assert_eq(out.data().get([0, 0, 0, 1]), 2.0)
  assert_eq(out.data().get([0, 0, 1, 0]), 3.0)
  assert_eq(out.data().get([0, 0, 1, 1]), 4.0)
}

///|
test "batchnorm2d known values" {
  // Input: N=1, C=2, H=1, W=1, values=[4.0, 6.0]
  // Ch0: (4-2)/sqrt(4+0) * 1 + 0 = 1.0
  // Ch1: (6-3)/sqrt(1+0) * 2 + 1 = 7.0
  let input = @autograd.Variable::new(
    @tensor.Tensor::from_array_with_shape([4.0, 6.0], [1, 2, 1, 1]),
  )
  let bn = @nn.BatchNorm2d::from_tensors(
    gamma=@tensor.Tensor::from_array([1.0, 2.0]),
    beta=@tensor.Tensor::from_array([0.0, 1.0]),
    mean=@tensor.Tensor::from_array([2.0, 3.0]),
    var_=@tensor.Tensor::from_array([4.0, 1.0]),
    eps=0.0,
  )
  let out = bn.forward(input)
  assert_eq(out.data().shape(), [1, 2, 1, 1])
  let tolerance = 1.0e-10
  assert_true((out.data().get([0, 0, 0, 0]) - 1.0).abs() < tolerance)
  assert_true((out.data().get([0, 1, 0, 0]) - 7.0).abs() < tolerance)
}

///|
test "layer enum dispatches forward" {
  // Linear layer
  let w = @tensor.Tensor::from_array_2d([[1.0], [1.0]])
  let b = @tensor.Tensor::from_array([0.0])
  let linear_layer : @nn.Layer = @nn.Layer::Linear(@nn.Linear::from_tensors(w, b))
  let x = @autograd.Variable::new(@tensor.Tensor::from_array_2d([[2.0, 3.0]]))
  let out1 = linear_layer.forward(x)
  assert_eq(out1.data().get([0, 0]), 5.0)

  // Relu layer
  let relu_layer : @nn.Layer = @nn.Layer::Relu
  let neg_input = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[-1.0, 2.0]]),
  )
  let out2 = relu_layer.forward(neg_input)
  assert_eq(out2.data().get([0, 0]), 0.0)
  assert_eq(out2.data().get([0, 1]), 2.0)

  // Flatten layer
  let flatten_layer : @nn.Layer = @nn.Layer::Flatten
  let input_4d = @autograd.Variable::new(
    @tensor.Tensor::from_array_with_shape(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 2, 2, 2],
    ),
  )
  let out3 = flatten_layer.forward(input_4d)
  assert_eq(out3.shape(), [1, 8])
  assert_eq(out3.data().get([0, 0]), 1.0)
  assert_eq(out3.data().get([0, 7]), 8.0)
}

///|
test "conv2d layer via layer enum" {
  let weight = @tensor.Tensor::from_array_with_shape([1.0], [1, 1, 1, 1])
  let conv = @nn.Conv2d::from_tensors(weight)
  let layer : @nn.Layer = @nn.Layer::Conv2d(conv)
  let input = @autograd.Variable::new(
    @tensor.Tensor::from_array_with_shape([5.0, 6.0, 7.0, 8.0], [1, 1, 2, 2]),
  )
  let out = layer.forward(input)
  assert_eq(out.data().get([0, 0, 0, 0]), 5.0)
}

///|
test "batchnorm2d layer via layer enum" {
  let bn = @nn.BatchNorm2d::from_tensors(
    gamma=@tensor.Tensor::from_array([1.0]),
    beta=@tensor.Tensor::from_array([0.0]),
    mean=@tensor.Tensor::from_array([0.0]),
    var_=@tensor.Tensor::from_array([1.0]),
    eps=0.0,
  )
  let layer : @nn.Layer = @nn.Layer::BatchNorm2d(bn)
  let input = @autograd.Variable::new(
    @tensor.Tensor::from_array_with_shape([3.0], [1, 1, 1, 1]),
  )
  let out = layer.forward(input)
  assert_eq(out.data().get([0, 0, 0, 0]), 3.0)
}
