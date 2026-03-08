///|
test "adam step updates single parameter" {
  let p = @autograd.Variable::new(
    @tensor.Tensor::from_array([4.0]),
    requires_grad=true,
  )
  // f = w^2, grad = 2w = 8.0
  let loss = p.mul(p).sum()
  loss.backward()
  let optimizer = Adam::new([p], lr=0.001)
  optimizer.step()
  // m = 0.1 * 8.0 = 0.8
  // v = 0.001 * 64.0 = 0.064
  // m_hat = 0.8 / 0.1 = 8.0
  // v_hat = 0.064 / 0.001 = 64.0
  // update = 0.001 * 8.0 / (sqrt(64.0) + 1e-8) = 0.001
  // new_w = 4.0 - 0.001 = 3.999
  let new_val = p.data().get([0])
  assert_true(new_val > 3.998)
  assert_true(new_val < 4.0)
}

///|
test "adam zero_grad clears gradients" {
  let p = @autograd.Variable::new(
    @tensor.Tensor::from_array([1.0]),
    requires_grad=true,
  )
  let loss = p.mul(p).sum()
  loss.backward()
  assert_true(p.grad() is Some(_))
  let optimizer = Adam::new([p])
  optimizer.zero_grad()
  assert_true(p.grad() is None)
}

///|
test "adam minimizes quadratic function" {
  // f(w) = (w - 3)^2, minimum at w = 3
  let w = @autograd.Variable::new(
    @tensor.Tensor::from_array([10.0]),
    requires_grad=true,
  )
  let target = @autograd.Variable::new(@tensor.Tensor::from_array([3.0]))
  let optimizer = Adam::new([w], lr=0.1)
  for _step = 0; _step < 200; _step = _step + 1 {
    let diff = w.sub(target)
    let loss = diff.mul(diff).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  }
  let final_w = w.data().get([0])
  assert_true(final_w > 2.9)
  assert_true(final_w < 3.1)
}

///|
test "adam training reduces mlp loss" {
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
  )
  let target = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[5.0], [7.0], [8.0]]),
  )
  let params = l1.parameters()
  params.append(l2.parameters())
  let optimizer = Adam::new(params, lr=0.01)
  // Initial loss
  let h0 = l1.forward(x).relu()
  let pred0 = l2.forward(h0)
  let diff0 = pred0.sub(target)
  let initial_loss = diff0.mul(diff0).sum().data().get([])
  // Train
  for _step = 0; _step < 100; _step = _step + 1 {
    let h = l1.forward(x).relu()
    let pred = l2.forward(h)
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  }
  let h_final = l1.forward(x).relu()
  let pred_final = l2.forward(h_final)
  let diff_final = pred_final.sub(target)
  let final_loss = diff_final.mul(diff_final).sum().data().get([])
  assert_true(final_loss < initial_loss)
}

///|
test "adam skips params without grad" {
  let p1 = @autograd.Variable::new(
    @tensor.Tensor::from_array([1.0]),
    requires_grad=true,
  )
  let p2 = @autograd.Variable::new(
    @tensor.Tensor::from_array([2.0]),
    requires_grad=true,
  )
  // Only p1 gets gradient
  let loss = p1.mul(p1).sum()
  loss.backward()
  let optimizer = Adam::new([p1, p2])
  optimizer.step()
  // p2 should remain unchanged
  assert_eq(p2.data().get([0]), 2.0)
}
