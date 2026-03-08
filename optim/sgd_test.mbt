///|
test "sgd step updates single parameter" {
  let p = @autograd.Variable::new(
    @tensor.Tensor::from_array([4.0]),
    requires_grad=true,
  )
  // Simulate gradient by backward on p*p
  // grad of p*p at p=4 is 8, but we want a controlled test.
  // Instead, manually set up: use mul + sum + backward to get grad = 2*p = 8
  // For a cleaner test, let's compute loss = p * const and backward.
  // Actually, let's just do: loss = p.mul(p).sum() → grad = 2*p = [8.0]
  // Then SGD with lr=0.1: new_p = 4.0 - 0.1*8.0 = 3.2
  let loss = p.mul(p).sum()
  loss.backward()
  // Verify grad is [8.0]
  assert_eq(p.grad().unwrap().get([0]), 8.0)
  let optimizer = @optim.SGD::new([p], lr=0.1)
  optimizer.step()
  // 4.0 - 0.1 * 8.0 = 3.2
  assert_eq(p.data().get([0]), 3.2)
}

///|
test "sgd step updates multiple parameters" {
  let p1 = @autograd.Variable::new(
    @tensor.Tensor::from_array([3.0]),
    requires_grad=true,
  )
  let p2 = @autograd.Variable::new(
    @tensor.Tensor::from_array([5.0]),
    requires_grad=true,
  )
  // loss = p1^2 + p2^2 → grad_p1 = 6.0, grad_p2 = 10.0
  let loss = p1.mul(p1).sum().add(p2.mul(p2).sum())
  loss.backward()
  let optimizer = @optim.SGD::new([p1, p2], lr=0.1)
  optimizer.step()
  // p1: 3.0 - 0.1*6.0 = 2.4
  // p2: 5.0 - 0.1*10.0 = 4.0
  assert_eq(p1.data().get([0]), 2.4)
  assert_eq(p2.data().get([0]), 4.0)
}

///|
test "sgd zero_grad clears all gradients" {
  let p1 = @autograd.Variable::new(
    @tensor.Tensor::from_array([1.0]),
    requires_grad=true,
  )
  let p2 = @autograd.Variable::new(
    @tensor.Tensor::from_array([2.0]),
    requires_grad=true,
  )
  let loss = p1.add(p2).sum()
  loss.backward()
  assert_true(p1.grad() is Some(_))
  assert_true(p2.grad() is Some(_))
  let optimizer = @optim.SGD::new([p1, p2])
  optimizer.zero_grad()
  assert_true(p1.grad() is None)
  assert_true(p2.grad() is None)
}

///|
test "sgd step skips parameter with no grad" {
  let p1 = @autograd.Variable::new(
    @tensor.Tensor::from_array([3.0]),
    requires_grad=true,
  )
  let p2 = @autograd.Variable::new(
    @tensor.Tensor::from_array([7.0]),
    requires_grad=true,
  )
  // Only compute grad for p1
  let loss = p1.mul(p1).sum()
  loss.backward()
  // p2 has no grad
  assert_true(p2.grad() is None)
  let optimizer = @optim.SGD::new([p1, p2], lr=0.1)
  optimizer.step()
  // p1 updated: 3.0 - 0.1*6.0 = 2.4
  assert_eq(p1.data().get([0]), 2.4)
  // p2 unchanged
  assert_eq(p2.data().get([0]), 7.0)
}

///|
test "sgd zero_grad on already none grad" {
  let p = @autograd.Variable::new(
    @tensor.Tensor::from_array([1.0]),
    requires_grad=true,
  )
  assert_true(p.grad() is None)
  let optimizer = @optim.SGD::new([p])
  optimizer.zero_grad() // should not panic
  assert_true(p.grad() is None)
}

///|
test "sgd default learning rate" {
  let p = @autograd.Variable::new(
    @tensor.Tensor::from_array([10.0]),
    requires_grad=true,
  )
  // loss = p^2 → grad = 20.0
  let loss = p.mul(p).sum()
  loss.backward()
  let optimizer = @optim.SGD::new([p]) // default lr = 0.01
  optimizer.step()
  // 10.0 - 0.01 * 20.0 = 9.8
  assert_eq(p.data().get([0]), 9.8)
}

///|
test "sgd training reduces quadratic loss" {
  // Minimize f(x) = (x - 3)^2, starting from x = 0
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([0.0]),
    requires_grad=true,
  )
  let target = @autograd.Variable::new(@tensor.Tensor::from_array([3.0]))
  let optimizer = @optim.SGD::new([x], lr=0.1)
  for _step = 0; _step < 100; _step = _step + 1 {
    let diff = x.sub(target)
    let loss = diff.mul(diff).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  }
  // x should be close to 3.0
  let final_x = x.data().get([0])
  assert_true(final_x > 2.9)
  assert_true(final_x < 3.1)
}

///|
test "sgd training linear regression" {
  let linear = @nn.Linear::new(2, 1, seed=42UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
  )
  let target = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[5.0], [7.0], [8.0]]),
  )
  let optimizer = @optim.SGD::new(linear.parameters(), lr=0.01)
  // Initial loss
  let init_output = linear.forward(x)
  let init_diff = init_output.sub(target)
  let init_loss = init_diff.mul(init_diff).sum()
  let initial_loss_val = init_loss.data().get([])
  // Train
  for _step = 0; _step < 50; _step = _step + 1 {
    let output = linear.forward(x)
    let diff = output.sub(target)
    let loss = diff.mul(diff).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  }
  // Final loss
  let final_output = linear.forward(x)
  let final_diff = final_output.sub(target)
  let final_loss = final_diff.mul(final_diff).sum()
  let final_loss_val = final_loss.data().get([])
  assert_true(final_loss_val < initial_loss_val)
}
