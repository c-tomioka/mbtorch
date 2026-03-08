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
