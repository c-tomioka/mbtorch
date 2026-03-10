///|
test "grad add scalar" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::scalar(3.0),
    requires_grad=true,
  )
  let y = @autograd.Variable::new(
    @tensor.Tensor::scalar(4.0),
    requires_grad=true,
  )
  let z = x.add(y)
  z.backward()
  assert_eq(x.grad().unwrap().get([]), 1.0)
  assert_eq(y.grad().unwrap().get([]), 1.0)
}

///|
test "grad mul scalar" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::scalar(3.0),
    requires_grad=true,
  )
  let y = @autograd.Variable::new(
    @tensor.Tensor::scalar(4.0),
    requires_grad=true,
  )
  let z = x.mul(y)
  z.backward()
  assert_eq(x.grad().unwrap().get([]), 4.0)
  assert_eq(y.grad().unwrap().get([]), 3.0)
}

///|
test "grad x squared" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::scalar(3.0),
    requires_grad=true,
  )
  let z = x.mul(x)
  z.backward()
  assert_eq(x.grad().unwrap().get([]), 6.0)
}

///|
test "grad chain add then mul" {
  // f(x,y) = (x+y)*x, x=2, y=3
  // df/dx = 2x+y = 7, df/dy = x = 2
  let x = @autograd.Variable::new(
    @tensor.Tensor::scalar(2.0),
    requires_grad=true,
  )
  let y = @autograd.Variable::new(
    @tensor.Tensor::scalar(3.0),
    requires_grad=true,
  )
  let s = x.add(y)
  let z = s.mul(x)
  z.backward()
  assert_eq(x.grad().unwrap().get([]), 7.0)
  assert_eq(y.grad().unwrap().get([]), 2.0)
}

///|
test "grad add 1d" {
  let a = @autograd.Variable::new(
    @tensor.Tensor::from_array([1.0, 2.0]),
    requires_grad=true,
  )
  let b = @autograd.Variable::new(
    @tensor.Tensor::from_array([3.0, 4.0]),
    requires_grad=true,
  )
  let c = a.add(b)
  c.backward()
  assert_eq(a.grad().unwrap().get([0]), 1.0)
  assert_eq(a.grad().unwrap().get([1]), 1.0)
  assert_eq(b.grad().unwrap().get([0]), 1.0)
  assert_eq(b.grad().unwrap().get([1]), 1.0)
}

///|
test "grad mul 1d" {
  let a = @autograd.Variable::new(
    @tensor.Tensor::from_array([2.0, 3.0]),
    requires_grad=true,
  )
  let b = @autograd.Variable::new(
    @tensor.Tensor::from_array([4.0, 5.0]),
    requires_grad=true,
  )
  let c = a.mul(b)
  c.backward()
  assert_eq(a.grad().unwrap().get([0]), 4.0)
  assert_eq(a.grad().unwrap().get([1]), 5.0)
  assert_eq(b.grad().unwrap().get([0]), 2.0)
  assert_eq(b.grad().unwrap().get([1]), 3.0)
}

///|
test "grad sum 1d" {
  let a = @autograd.Variable::new(
    @tensor.Tensor::from_array([1.0, 2.0, 3.0]),
    requires_grad=true,
  )
  let c = a.sum()
  c.backward()
  assert_eq(a.grad().unwrap().get([0]), 1.0)
  assert_eq(a.grad().unwrap().get([1]), 1.0)
  assert_eq(a.grad().unwrap().get([2]), 1.0)
}

///|
test "grad sum of mul 1d" {
  // c = sum(a*b), dc/da = b, dc/db = a
  let a = @autograd.Variable::new(
    @tensor.Tensor::from_array([2.0, 3.0]),
    requires_grad=true,
  )
  let b = @autograd.Variable::new(
    @tensor.Tensor::from_array([4.0, 5.0]),
    requires_grad=true,
  )
  let prod = a.mul(b)
  let c = prod.sum()
  c.backward()
  assert_eq(a.grad().unwrap().get([0]), 4.0)
  assert_eq(a.grad().unwrap().get([1]), 5.0)
  assert_eq(b.grad().unwrap().get([0]), 2.0)
  assert_eq(b.grad().unwrap().get([1]), 3.0)
}

///|
test "grad matmul 2x2" {
  // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
  // C = A @ B = [[19,22],[43,50]]
  // dC = ones(2,2) = [[1,1],[1,1]]
  // dA = dC @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
  // dB = A^T @ dC = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
  let a = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 2.0], [3.0, 4.0]]),
    requires_grad=true,
  )
  let b = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[5.0, 6.0], [7.0, 8.0]]),
    requires_grad=true,
  )
  let c = a.matmul(b)
  c.backward()
  let ga = a.grad().unwrap()
  assert_eq(ga.get([0, 0]), 11.0)
  assert_eq(ga.get([0, 1]), 15.0)
  assert_eq(ga.get([1, 0]), 11.0)
  assert_eq(ga.get([1, 1]), 15.0)
  let gb = b.grad().unwrap()
  assert_eq(gb.get([0, 0]), 4.0)
  assert_eq(gb.get([0, 1]), 4.0)
  assert_eq(gb.get([1, 0]), 6.0)
  assert_eq(gb.get([1, 1]), 6.0)
}

///|
test "grad accumulates for shared input" {
  // f(x) = x * x, df/dx = 2x
  let x = @autograd.Variable::new(
    @tensor.Tensor::scalar(5.0),
    requires_grad=true,
  )
  let z = x.mul(x)
  z.backward()
  assert_eq(x.grad().unwrap().get([]), 10.0)
}

///|
test "no grad for non-requires-grad" {
  let x = @autograd.Variable::new(@tensor.Tensor::scalar(3.0))
  let y = @autograd.Variable::new(
    @tensor.Tensor::scalar(4.0),
    requires_grad=true,
  )
  let z = x.add(y)
  z.backward()
  assert_true(x.grad() is None)
  assert_eq(y.grad().unwrap().get([]), 1.0)
}

///|
test "zero_grad resets gradient" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::scalar(3.0),
    requires_grad=true,
  )
  let y = @autograd.Variable::new(
    @tensor.Tensor::scalar(4.0),
    requires_grad=true,
  )
  let z = x.mul(y)
  z.backward()
  assert_eq(x.grad().unwrap().get([]), 4.0)
  x.zero_grad()
  assert_true(x.grad() is None)
}

///|
test "grad_check add" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::scalar(3.0),
    requires_grad=true,
  )
  let y = @autograd.Variable::new(
    @tensor.Tensor::scalar(4.0),
    requires_grad=true,
  )
  assert_true(
    @autograd.grad_check(fn(inputs) { inputs[0].add(inputs[1]).sum() }, [x, y]),
  )
}

///|
test "grad_check mul" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([2.0, 3.0]),
    requires_grad=true,
  )
  let y = @autograd.Variable::new(
    @tensor.Tensor::from_array([4.0, 5.0]),
    requires_grad=true,
  )
  assert_true(
    @autograd.grad_check(fn(inputs) { inputs[0].mul(inputs[1]).sum() }, [x, y]),
  )
}

///|
test "grad_check matmul" {
  let a = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 2.0], [3.0, 4.0]]),
    requires_grad=true,
  )
  let b = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[5.0, 6.0], [7.0, 8.0]]),
    requires_grad=true,
  )
  assert_true(
    @autograd.grad_check(fn(inputs) { inputs[0].matmul(inputs[1]).sum() }, [
      a, b,
    ]),
  )
}

///|
test "grad_check composite" {
  // f(x,y) = sum((x*y) + x)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([2.0, 3.0]),
    requires_grad=true,
  )
  let y = @autograd.Variable::new(
    @tensor.Tensor::from_array([4.0, 5.0]),
    requires_grad=true,
  )
  assert_true(
    @autograd.grad_check(
      fn(inputs) { inputs[0].mul(inputs[1]).add(inputs[0]).sum() },
      [x, y],
    ),
  )
}

///|
test "grad sub scalar" {
  // f(x,y) = x - y, df/dx = 1, df/dy = -1
  let x = @autograd.Variable::new(
    @tensor.Tensor::scalar(5.0),
    requires_grad=true,
  )
  let y = @autograd.Variable::new(
    @tensor.Tensor::scalar(3.0),
    requires_grad=true,
  )
  let z = x.sub(y)
  z.backward()
  assert_eq(x.grad().unwrap().get([]), 1.0)
  assert_eq(y.grad().unwrap().get([]), -1.0)
}

///|
test "grad sub 1d" {
  let a = @autograd.Variable::new(
    @tensor.Tensor::from_array([5.0, 3.0]),
    requires_grad=true,
  )
  let b = @autograd.Variable::new(
    @tensor.Tensor::from_array([1.0, 2.0]),
    requires_grad=true,
  )
  let c = a.sub(b).sum()
  c.backward()
  assert_eq(a.grad().unwrap().get([0]), 1.0)
  assert_eq(a.grad().unwrap().get([1]), 1.0)
  assert_eq(b.grad().unwrap().get([0]), -1.0)
  assert_eq(b.grad().unwrap().get([1]), -1.0)
}

///|
test "grad relu positive" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([2.0, 3.0]),
    requires_grad=true,
  )
  let y = x.relu().sum()
  y.backward()
  assert_eq(x.grad().unwrap().get([0]), 1.0)
  assert_eq(x.grad().unwrap().get([1]), 1.0)
}

///|
test "grad relu mixed" {
  // relu([2, -1, 0, 3]) = [2, 0, 0, 3]
  // grad: [1, 0, 0, 1]
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([2.0, -1.0, 0.0, 3.0]),
    requires_grad=true,
  )
  let y = x.relu().sum()
  y.backward()
  assert_eq(x.grad().unwrap().get([0]), 1.0)
  assert_eq(x.grad().unwrap().get([1]), 0.0)
  assert_eq(x.grad().unwrap().get([2]), 0.0)
  assert_eq(x.grad().unwrap().get([3]), 1.0)
}

///|
test "grad_check sub" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([2.0, 3.0]),
    requires_grad=true,
  )
  let y = @autograd.Variable::new(
    @tensor.Tensor::from_array([4.0, 5.0]),
    requires_grad=true,
  )
  assert_true(
    @autograd.grad_check(fn(inputs) { inputs[0].sub(inputs[1]).sum() }, [x, y]),
  )
}

///|
test "grad_check relu" {
  // Use positive values to avoid discontinuity at 0
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([1.0, 2.0, 3.0]),
    requires_grad=true,
  )
  assert_true(@autograd.grad_check(fn(inputs) { inputs[0].relu().sum() }, [x]))
}

///|
test "set_data updates variable" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::scalar(3.0),
    requires_grad=true,
  )
  assert_eq(x.data().get([]), 3.0)
  x.set_data(@tensor.Tensor::scalar(7.0))
  assert_eq(x.data().get([]), 7.0)
}

///|
test "tanh forward values" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([0.0, 100.0, -100.0]),
  )
  let y = x.tanh()
  let d = y.data()
  // tanh(0) = 0
  assert_true(d.get([0]).abs() < 1.0e-10)
  // tanh(100) ≈ 1
  assert_true(d.get([1]) > 0.999)
  // tanh(-100) ≈ -1
  assert_true(d.get([2]) < -0.999)
}

///|
test "sigmoid forward values" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([0.0, 100.0, -100.0]),
  )
  let y = x.sigmoid()
  let d = y.data()
  // sigmoid(0) = 0.5
  assert_true((d.get([0]) - 0.5).abs() < 1.0e-10)
  // sigmoid(100) ≈ 1
  assert_true(d.get([1]) > 0.999)
  // sigmoid(-100) ≈ 0
  assert_true(d.get([2]) < 0.001)
}

///|
test "tanh backward manual" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([0.5]),
    requires_grad=true,
  )
  let y = x.tanh().sum()
  y.backward()
  // d/dx tanh(0.5) = 1 - tanh(0.5)^2 ≈ 0.7864
  let grad_val = x.grad().unwrap().get([0])
  assert_true((grad_val - 0.7864).abs() < 0.001)
}

///|
test "sigmoid backward manual" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([1.0]),
    requires_grad=true,
  )
  let y = x.sigmoid().sum()
  y.backward()
  // sigmoid(1) ≈ 0.7311, d/dx = 0.7311 * (1 - 0.7311) ≈ 0.1966
  let grad_val = x.grad().unwrap().get([0])
  assert_true((grad_val - 0.1966).abs() < 0.001)
}

///|
test "grad_check tanh" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([0.0, 0.5, -1.0, 2.0]),
    requires_grad=true,
  )
  assert_true(@autograd.grad_check(fn(inputs) { inputs[0].tanh().sum() }, [x]))
}

///|
test "grad_check sigmoid" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([0.0, 1.0, -1.0, 3.0]),
    requires_grad=true,
  )
  assert_true(
    @autograd.grad_check(fn(inputs) { inputs[0].sigmoid().sum() }, [x]),
  )
}

///|
test "grad_check scale" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([1.0, 2.0, -3.0, 0.5]),
    requires_grad=true,
  )
  assert_true(
    @autograd.grad_check(fn(inputs) { inputs[0].scale(0.5).sum() }, [x]),
  )
}

///|
test "grad_check softmax" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]]),
    requires_grad=true,
  )
  // softmax().sum() is constant 1.0, so use non-uniform weights for non-trivial gradients
  let w = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0], [0.5, 1.5, 2.0]]),
  )
  assert_true(
    @autograd.grad_check(
      fn(inputs) { inputs[0].softmax().mul(w).sum() },
      [x],
    ),
  )
}

///|
test "grad_check reshape" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    requires_grad=true,
  )
  assert_true(
    @autograd.grad_check(
      fn(inputs) { inputs[0].reshape([3, 2]).sum() },
      [x],
    ),
  )
}

///|
test "grad_check transpose_last2" {
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_with_shape(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
      [2, 2, 3],
    ),
    requires_grad=true,
  )
  assert_true(
    @autograd.grad_check(
      fn(inputs) { inputs[0].transpose_last2().sum() },
      [x],
    ),
  )
}

///|
test "grad_check batched_matmul_2d" {
  let a = @autograd.Variable::new(
    @tensor.Tensor::from_array_with_shape(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      [2, 1, 3],
    ),
    requires_grad=true,
  )
  let b = @autograd.Variable::new(
    @tensor.Tensor::from_array_with_shape(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      [2, 3, 1],
    ),
    requires_grad=true,
  )
  assert_true(
    @autograd.grad_check(
      fn(inputs) { inputs[0].batched_matmul_2d(inputs[1]).sum() },
      [a, b],
    ),
  )
}
