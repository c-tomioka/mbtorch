///|
test "reshape preserves data" {
  let t = @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  // 2x3 -> 6
  let r1 = t.reshape([6])
  assert_eq(r1.shape(), [6])
  assert_eq(r1.get([0]), 1.0)
  assert_eq(r1.get([5]), 6.0)
  // 6 -> 3x2
  let r2 = r1.reshape([3, 2])
  assert_eq(r2.shape(), [3, 2])
  assert_eq(r2.get([0, 0]), 1.0)
  assert_eq(r2.get([2, 1]), 6.0)
  // 3x2 -> 1x1x2x3
  let r3 = r2.reshape([1, 1, 2, 3])
  assert_eq(r3.shape(), [1, 1, 2, 3])
  assert_eq(r3.get([0, 0, 0, 0]), 1.0)
  assert_eq(r3.get([0, 0, 1, 2]), 6.0)
}

///|
test "conv2d 1x1 kernel" {
  // Input: N=1, C=1, H=2, W=2
  let input = @tensor.Tensor::from_array_with_shape(
    [1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2],
  )
  // Kernel: out_ch=1, in_ch=1, kH=1, kW=1
  let kernel = @tensor.Tensor::from_array_with_shape([2.0], [1, 1, 1, 1])
  let output = input.conv2d(kernel, stride=[1, 1], pad=[0, 0, 0, 0])
  assert_eq(output.shape(), [1, 1, 2, 2])
  assert_eq(output.get([0, 0, 0, 0]), 2.0)
  assert_eq(output.get([0, 0, 0, 1]), 4.0)
  assert_eq(output.get([0, 0, 1, 0]), 6.0)
  assert_eq(output.get([0, 0, 1, 1]), 8.0)
}

///|
test "conv2d 3x3 kernel no padding" {
  // Input: N=1, C=1, H=4, W=4 (values 1..16)
  let data : Array[Double] = []
  for i = 1; i <= 16; i = i + 1 {
    data.push(i.to_double())
  }
  let input = @tensor.Tensor::from_array_with_shape(data, [1, 1, 4, 4])
  // Kernel: all ones, 3x3
  let kernel = @tensor.Tensor::from_array_with_shape(
    Array::make(9, 1.0), [1, 1, 3, 3],
  )
  let output = input.conv2d(kernel, stride=[1, 1], pad=[0, 0, 0, 0])
  // Output: [1, 1, 2, 2]
  assert_eq(output.shape(), [1, 1, 2, 2])
  // Top-left 3x3 sum: 1+2+3+5+6+7+9+10+11 = 54
  assert_eq(output.get([0, 0, 0, 0]), 54.0)
  // Top-right: 2+3+4+6+7+8+10+11+12 = 63
  assert_eq(output.get([0, 0, 0, 1]), 63.0)
  // Bottom-left: 5+6+7+9+10+11+13+14+15 = 90
  assert_eq(output.get([0, 0, 1, 0]), 90.0)
  // Bottom-right: 6+7+8+10+11+12+14+15+16 = 99
  assert_eq(output.get([0, 0, 1, 1]), 99.0)
}

///|
test "conv2d 3x3 kernel with padding" {
  // Input: N=1, C=1, H=2, W=2, values=[1,2,3,4]
  let input = @tensor.Tensor::from_array_with_shape(
    [1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2],
  )
  // Kernel: all ones, 3x3, pad=1 on all sides
  let kernel = @tensor.Tensor::from_array_with_shape(
    Array::make(9, 1.0), [1, 1, 3, 3],
  )
  let output = input.conv2d(kernel, stride=[1, 1], pad=[1, 1, 1, 1])
  // Output: [1, 1, 2, 2] (same spatial size)
  assert_eq(output.shape(), [1, 1, 2, 2])
  // [0,0]: pad region, only 1+2+3+4 from bottom-right quadrant
  // Actually: 3x3 centered at (0,0) with pad=1:
  // row -1: 0 0 0
  // row  0: 0 1 2
  // row  1: 0 3 4
  // sum = 1+2+3+4 = 10
  assert_eq(output.get([0, 0, 0, 0]), 10.0)
}

///|
test "conv2d multi-channel" {
  // Input: N=1, C=2, H=2, W=2
  // Ch0: [1,2,3,4], Ch1: [5,6,7,8]
  let input = @tensor.Tensor::from_array_with_shape(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 2, 2, 2],
  )
  // Kernel: out_ch=1, in_ch=2, kH=1, kW=1
  // Filter: ch0_weight=1, ch1_weight=2
  let kernel = @tensor.Tensor::from_array_with_shape(
    [1.0, 2.0], [1, 2, 1, 1],
  )
  let output = input.conv2d(kernel, stride=[1, 1], pad=[0, 0, 0, 0])
  assert_eq(output.shape(), [1, 1, 2, 2])
  // [0,0]: 1*1 + 2*5 = 11
  assert_eq(output.get([0, 0, 0, 0]), 11.0)
  // [0,1]: 1*2 + 2*6 = 14
  assert_eq(output.get([0, 0, 0, 1]), 14.0)
  // [1,0]: 1*3 + 2*7 = 17
  assert_eq(output.get([0, 0, 1, 0]), 17.0)
  // [1,1]: 1*4 + 2*8 = 20
  assert_eq(output.get([0, 0, 1, 1]), 20.0)
}

///|
test "conv2d stride 2" {
  // Input: N=1, C=1, H=4, W=4
  let data : Array[Double] = []
  for i = 1; i <= 16; i = i + 1 {
    data.push(i.to_double())
  }
  let input = @tensor.Tensor::from_array_with_shape(data, [1, 1, 4, 4])
  // Kernel: 1x1 identity
  let kernel = @tensor.Tensor::from_array_with_shape([1.0], [1, 1, 1, 1])
  let output = input.conv2d(kernel, stride=[2, 2], pad=[0, 0, 0, 0])
  // Output: [1, 1, 2, 2]
  assert_eq(output.shape(), [1, 1, 2, 2])
  assert_eq(output.get([0, 0, 0, 0]), 1.0) // (0,0)
  assert_eq(output.get([0, 0, 0, 1]), 3.0) // (0,2)
  assert_eq(output.get([0, 0, 1, 0]), 9.0) // (2,0)
  assert_eq(output.get([0, 0, 1, 1]), 11.0) // (2,2)
}

///|
test "conv2d with bias" {
  let input = @tensor.Tensor::from_array_with_shape(
    [1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2],
  )
  let kernel = @tensor.Tensor::from_array_with_shape([1.0], [1, 1, 1, 1])
  let bias = @tensor.Tensor::from_array([10.0])
  let output = input.conv2d(
    kernel, bias=Some(bias), stride=[1, 1], pad=[0, 0, 0, 0],
  )
  assert_eq(output.get([0, 0, 0, 0]), 11.0)
  assert_eq(output.get([0, 0, 0, 1]), 12.0)
  assert_eq(output.get([0, 0, 1, 0]), 13.0)
  assert_eq(output.get([0, 0, 1, 1]), 14.0)
}
