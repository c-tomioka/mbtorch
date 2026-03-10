///|
test "scalar creates 0-dim tensor" {
  let s = @tensor.Tensor::scalar(3.0)
  assert_eq(s.shape(), [])
  assert_eq(s.ndim(), 0)
  assert_eq(s.size(), 1)
  assert_eq(s.get([]), 3.0)
}

///|
test "from_array creates 1d tensor" {
  let v = @tensor.Tensor::from_array([1.0, 2.0, 3.0])
  assert_eq(v.shape(), [3])
  assert_eq(v.ndim(), 1)
  assert_eq(v.size(), 3)
  assert_eq(v.get([0]), 1.0)
  assert_eq(v.get([2]), 3.0)
}

///|
test "from_array_2d creates 2d tensor" {
  let m = @tensor.Tensor::from_array_2d([[1.0, 2.0], [3.0, 4.0]])
  assert_eq(m.shape(), [2, 2])
  assert_eq(m.ndim(), 2)
  assert_eq(m.size(), 4)
  assert_eq(m.get([0, 1]), 2.0)
  assert_eq(m.get([1, 0]), 3.0)
}

///|
test "zeros creates all-zero tensor" {
  let z = @tensor.Tensor::zeros([2, 3])
  assert_eq(z.shape(), [2, 3])
  assert_eq(z.size(), 6)
  assert_eq(z.get([0, 0]), 0.0)
  assert_eq(z.get([1, 2]), 0.0)
}

///|
test "ones creates all-one tensor" {
  let o = @tensor.Tensor::ones([3])
  assert_eq(o.shape(), [3])
  assert_eq(o.size(), 3)
  assert_eq(o.get([0]), 1.0)
  assert_eq(o.get([2]), 1.0)
}

///|
test "get retrieves correct element 1d" {
  let v = @tensor.Tensor::from_array([10.0, 20.0, 30.0])
  assert_eq(v.get([1]), 20.0)
}

///|
test "get retrieves correct element 2d" {
  let m = @tensor.Tensor::from_array_2d([[1.0, 2.0], [3.0, 4.0]])
  assert_eq(m.get([1, 0]), 3.0)
}

///|
test "add element-wise 1d" {
  let a = @tensor.Tensor::from_array([1.0, 2.0, 3.0])
  let b = @tensor.Tensor::from_array([4.0, 5.0, 6.0])
  let c = a.add(b)
  assert_eq(c.shape(), [3])
  assert_eq(c.get([0]), 5.0)
  assert_eq(c.get([1]), 7.0)
  assert_eq(c.get([2]), 9.0)
}

///|
test "add element-wise 2d" {
  let a = @tensor.Tensor::from_array_2d([[1.0, 2.0], [3.0, 4.0]])
  let b = @tensor.Tensor::from_array_2d([[5.0, 6.0], [7.0, 8.0]])
  let c = a.add(b)
  assert_eq(c.shape(), [2, 2])
  assert_eq(c.get([0, 0]), 6.0)
  assert_eq(c.get([1, 1]), 12.0)
}

///|
test "mul element-wise 1d" {
  let a = @tensor.Tensor::from_array([1.0, 2.0, 3.0])
  let b = @tensor.Tensor::from_array([4.0, 5.0, 6.0])
  let c = a.mul(b)
  assert_eq(c.shape(), [3])
  assert_eq(c.get([0]), 4.0)
  assert_eq(c.get([1]), 10.0)
  assert_eq(c.get([2]), 18.0)
}

///|
test "operator overload add" {
  let a = @tensor.Tensor::from_array([1.0, 2.0, 3.0])
  let b = @tensor.Tensor::from_array([4.0, 5.0, 6.0])
  let c = a + b
  assert_eq(c.get([0]), 5.0)
  assert_eq(c.get([1]), 7.0)
  assert_eq(c.get([2]), 9.0)
}

///|
test "operator overload mul" {
  let a = @tensor.Tensor::from_array([1.0, 2.0, 3.0])
  let b = @tensor.Tensor::from_array([4.0, 5.0, 6.0])
  let c = a * b
  assert_eq(c.get([0]), 4.0)
  assert_eq(c.get([1]), 10.0)
  assert_eq(c.get([2]), 18.0)
}

///|
test "matmul 2x2 * 2x2" {
  let a = @tensor.Tensor::from_array_2d([[1.0, 2.0], [3.0, 4.0]])
  let b = @tensor.Tensor::from_array_2d([[5.0, 6.0], [7.0, 8.0]])
  let c = a.matmul(b)
  assert_eq(c.shape(), [2, 2])
  assert_eq(c.get([0, 0]), 19.0)
  assert_eq(c.get([0, 1]), 22.0)
  assert_eq(c.get([1, 0]), 43.0)
  assert_eq(c.get([1, 1]), 50.0)
}

///|
test "matmul 2x3 * 3x2" {
  let a = @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let b = @tensor.Tensor::from_array_2d([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
  let c = a.matmul(b)
  assert_eq(c.shape(), [2, 2])
  assert_eq(c.get([0, 0]), 58.0)
  assert_eq(c.get([0, 1]), 64.0)
  assert_eq(c.get([1, 0]), 139.0)
  assert_eq(c.get([1, 1]), 154.0)
}

///|
test "matmul 1x3 * 3x1" {
  let a = @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0]])
  let b = @tensor.Tensor::from_array_2d([[4.0], [5.0], [6.0]])
  let c = a.matmul(b)
  assert_eq(c.shape(), [1, 1])
  assert_eq(c.get([0, 0]), 32.0)
}

///|
test "show scalar" {
  let s = @tensor.Tensor::scalar(3.0)
  let str = s.to_string()
  assert_true(str.contains("3"))
}

///|
test "show 1d" {
  let v = @tensor.Tensor::from_array([1.0, 2.0, 3.0])
  let str = v.to_string()
  assert_true(str.contains("1"))
  assert_true(str.contains("3"))
}

///|
test "transpose 2x3" {
  let m = @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let t = m.transpose()
  assert_eq(t.shape(), [3, 2])
  assert_eq(t.get([0, 0]), 1.0)
  assert_eq(t.get([0, 1]), 4.0)
  assert_eq(t.get([1, 0]), 2.0)
  assert_eq(t.get([1, 1]), 5.0)
  assert_eq(t.get([2, 0]), 3.0)
  assert_eq(t.get([2, 1]), 6.0)
}

///|
test "transpose 1x1" {
  let m = @tensor.Tensor::from_array_2d([[5.0]])
  let t = m.transpose()
  assert_eq(t.shape(), [1, 1])
  assert_eq(t.get([0, 0]), 5.0)
}

///|
test "sum 1d" {
  let v = @tensor.Tensor::from_array([1.0, 2.0, 3.0])
  let s = v.sum()
  assert_eq(s.shape(), [])
  assert_eq(s.get([]), 6.0)
}

///|
test "sum 2d" {
  let m = @tensor.Tensor::from_array_2d([[1.0, 2.0], [3.0, 4.0]])
  let s = m.sum()
  assert_eq(s.shape(), [])
  assert_eq(s.get([]), 10.0)
}

///|
test "sum scalar" {
  let s = @tensor.Tensor::scalar(5.0)
  let r = s.sum()
  assert_eq(r.shape(), [])
  assert_eq(r.get([]), 5.0)
}

///|
test "sub element-wise 1d" {
  let a = @tensor.Tensor::from_array([5.0, 3.0, 1.0])
  let b = @tensor.Tensor::from_array([1.0, 2.0, 3.0])
  let c = a.sub(b)
  assert_eq(c.shape(), [3])
  assert_eq(c.get([0]), 4.0)
  assert_eq(c.get([1]), 1.0)
  assert_eq(c.get([2]), -2.0)
}

///|
test "sub element-wise 2d" {
  let a = @tensor.Tensor::from_array_2d([[5.0, 6.0], [7.0, 8.0]])
  let b = @tensor.Tensor::from_array_2d([[1.0, 2.0], [3.0, 4.0]])
  let c = a.sub(b)
  assert_eq(c.shape(), [2, 2])
  assert_eq(c.get([0, 0]), 4.0)
  assert_eq(c.get([1, 1]), 4.0)
}

///|
test "scale multiplies all elements" {
  let a = @tensor.Tensor::from_array([2.0, -3.0, 4.0])
  let b = a.scale(0.5)
  assert_eq(b.shape(), [3])
  assert_eq(b.get([0]), 1.0)
  assert_eq(b.get([1]), -1.5)
  assert_eq(b.get([2]), 2.0)
}

///|
test "rand produces correct shape" {
  let t = @tensor.Tensor::rand([2, 3])
  assert_eq(t.shape(), [2, 3])
  assert_eq(t.size(), 6)
}

///|
test "rand seed reproducibility" {
  let a = @tensor.Tensor::rand([3], seed=42UL)
  let b = @tensor.Tensor::rand([3], seed=42UL)
  assert_eq(a.get([0]), b.get([0]))
  assert_eq(a.get([1]), b.get([1]))
  assert_eq(a.get([2]), b.get([2]))
}

///|
test "to_array returns flat data copy" {
  let t = @tensor.Tensor::from_array_2d([[1.0, 2.0], [3.0, 4.0]])
  let arr = t.to_array()
  assert_eq(arr.length(), 4)
  assert_eq(arr[0], 1.0)
  assert_eq(arr[1], 2.0)
  assert_eq(arr[2], 3.0)
  assert_eq(arr[3], 4.0)
}

///|
test "from_array_with_shape creates correct tensor" {
  let t = @tensor.Tensor::from_array_with_shape(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    [2, 3],
  )
  assert_eq(t.shape(), [2, 3])
  assert_eq(t.get([0, 0]), 1.0)
  assert_eq(t.get([0, 2]), 3.0)
  assert_eq(t.get([1, 0]), 4.0)
  assert_eq(t.get([1, 2]), 6.0)
}

///|
test "tensor div computes element-wise division" {
  let a = @tensor.Tensor::from_array([10.0, 9.0, 8.0])
  let b = @tensor.Tensor::from_array([2.0, 3.0, 4.0])
  let result = a.div(b)
  assert_eq(result.get([0]), 5.0)
  assert_eq(result.get([1]), 3.0)
  assert_eq(result.get([2]), 2.0)
}

///|
test "tensor sqrt computes element-wise square root" {
  let t = @tensor.Tensor::from_array([4.0, 9.0, 16.0, 0.0])
  let result = t.sqrt()
  assert_eq(result.shape(), [4])
  assert_eq(result.get([0]), 2.0)
  assert_eq(result.get([1]), 3.0)
  assert_eq(result.get([2]), 4.0)
  assert_eq(result.get([3]), 0.0)
}

///|
test "exp basic values" {
  let t = @tensor.Tensor::from_array([0.0, 1.0, -1.0])
  let result = t.exp()
  assert_eq(result.shape(), [3])
  // exp(0) = 1.0
  assert_true((result.get([0]) - 1.0).abs() < 1.0e-10)
  // exp(1) ≈ 2.71828
  assert_true((result.get([1]) - 2.718281828459045).abs() < 1.0e-10)
  // exp(-1) ≈ 0.36788
  assert_true((result.get([2]) - 0.36787944117144233).abs() < 1.0e-10)
}

///|
test "softmax 1d" {
  let t = @tensor.Tensor::from_array([1.0, 2.0, 3.0])
  let result = t.softmax()
  assert_eq(result.shape(), [3])
  // Sum should be 1.0
  let sum = result.get([0]) + result.get([1]) + result.get([2])
  assert_true((sum - 1.0).abs() < 1.0e-10)
  // Monotonically increasing: softmax(1) < softmax(2) < softmax(3)
  assert_true(result.get([0]) < result.get([1]))
  assert_true(result.get([1]) < result.get([2]))
}

///|
test "softmax 2d row-wise" {
  let t = @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
  let result = t.softmax()
  assert_eq(result.shape(), [2, 3])
  // Row 0 sums to 1
  let row0_sum = result.get([0, 0]) + result.get([0, 1]) + result.get([0, 2])
  assert_true((row0_sum - 1.0).abs() < 1.0e-10)
  // Row 1 sums to 1, and all equal
  let row1_sum = result.get([1, 0]) + result.get([1, 1]) + result.get([1, 2])
  assert_true((row1_sum - 1.0).abs() < 1.0e-10)
  assert_true((result.get([1, 0]) - result.get([1, 1])).abs() < 1.0e-10)
}

///|
test "softmax numerical stability" {
  // Large values should not overflow
  let t = @tensor.Tensor::from_array([1000.0, 1001.0, 1002.0])
  let result = t.softmax()
  let sum = result.get([0]) + result.get([1]) + result.get([2])
  assert_true((sum - 1.0).abs() < 1.0e-10)
  // Should be same relative distribution as softmax([0, 1, 2])
  assert_true(result.get([0]) < result.get([1]))
  assert_true(result.get([1]) < result.get([2]))
}

///|
test "softmax 3d last axis" {
  // [2, 2, 3] tensor
  let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
  let t = @tensor.Tensor::from_array_with_shape(data, [2, 2, 3])
  let result = t.softmax()
  assert_eq(result.shape(), [2, 2, 3])
  // Each "row" of 3 elements along last axis sums to 1.0
  let eps = 1.0e-10
  let r = result.to_array()
  // Row 0: indices 0,1,2
  assert_true((r[0] + r[1] + r[2] - 1.0).abs() < eps)
  // Row 1: indices 3,4,5
  assert_true((r[3] + r[4] + r[5] - 1.0).abs() < eps)
  // Row 2: indices 6,7,8
  assert_true((r[6] + r[7] + r[8] - 1.0).abs() < eps)
  // Row 3: indices 9,10,11
  assert_true((r[9] + r[10] + r[11] - 1.0).abs() < eps)
}

///|
test "batched_matmul_2d basic" {
  // Batch 0: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
  // Batch 1: [[1,0],[0,1]] @ [[2,3],[4,5]] = [[2,3],[4,5]]
  let a_data = [1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0]
  let a = @tensor.Tensor::from_array_with_shape(a_data, [2, 2, 2])
  let b_data = [5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0, 5.0]
  let b = @tensor.Tensor::from_array_with_shape(b_data, [2, 2, 2])
  let result = a.batched_matmul_2d(b)
  assert_eq(result.shape(), [2, 2, 2])
  let r = result.to_array()
  // Batch 0
  assert_eq(r[0], 19.0)
  assert_eq(r[1], 22.0)
  assert_eq(r[2], 43.0)
  assert_eq(r[3], 50.0)
  // Batch 1 (identity @ matrix = matrix)
  assert_eq(r[4], 2.0)
  assert_eq(r[5], 3.0)
  assert_eq(r[6], 4.0)
  assert_eq(r[7], 5.0)
}

///|
test "transpose_last2 3d" {
  // [2, 3, 2] -> [2, 2, 3]
  let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
  let t = @tensor.Tensor::from_array_with_shape(data, [2, 3, 2])
  let result = t.transpose_last2()
  assert_eq(result.shape(), [2, 2, 3])
  let r = result.to_array()
  // Batch 0: [[1,2],[3,4],[5,6]] transposed = [[1,3,5],[2,4,6]]
  assert_eq(r[0], 1.0)
  assert_eq(r[1], 3.0)
  assert_eq(r[2], 5.0)
  assert_eq(r[3], 2.0)
  assert_eq(r[4], 4.0)
  assert_eq(r[5], 6.0)
  // Batch 1: [[7,8],[9,10],[11,12]] transposed = [[7,9,11],[8,10,12]]
  assert_eq(r[6], 7.0)
  assert_eq(r[7], 9.0)
  assert_eq(r[8], 11.0)
  assert_eq(r[9], 8.0)
  assert_eq(r[10], 10.0)
  assert_eq(r[11], 12.0)
}
