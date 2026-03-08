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
