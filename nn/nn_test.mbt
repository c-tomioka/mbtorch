///|
test "linear forward output shape" {
  let linear = @nn.Linear::new(3, 2)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
  )
  let output = linear.forward(x)
  assert_eq(output.shape(), [2, 2])
}

///|
test "linear forward known weights" {
  // weight [2, 3], bias [3]
  // input [1, 2] @ weight [2, 3] = [1, 3], + bias [3]
  let linear = @nn.Linear::new(2, 3)
  // Set known weights: weight = [[1, 0, -1], [0, 1, 2]], bias = [0.1, 0.2, 0.3]
  linear.weight.set_data(
    @tensor.Tensor::from_array_2d([[1.0, 0.0, -1.0], [0.0, 1.0, 2.0]]),
  )
  linear.bias.set_data(@tensor.Tensor::from_array([0.1, 0.2, 0.3]))
  let x = @autograd.Variable::new(@tensor.Tensor::from_array_2d([[2.0, 3.0]]))
  // output = [2,3] @ [[1,0,-1],[0,1,2]] + [0.1,0.2,0.3]
  //        = [2*1+3*0, 2*0+3*1, 2*(-1)+3*2] + [0.1, 0.2, 0.3]
  //        = [2, 3, 4] + [0.1, 0.2, 0.3]
  //        = [2.1, 3.2, 4.3]
  let output = linear.forward(x)
  assert_eq(output.shape(), [1, 3])
  assert_eq(output.data().get([0, 0]), 2.1)
  assert_eq(output.data().get([0, 1]), 3.2)
  assert_eq(output.data().get([0, 2]), 4.3)
}

///|
test "linear parameters count" {
  let linear = @nn.Linear::new(4, 3)
  let params = linear.parameters()
  assert_eq(params.length(), 2)
}

///|
test "linear parameters require grad" {
  let linear = @nn.Linear::new(4, 3)
  let params = linear.parameters()
  for i = 0; i < params.length(); i = i + 1 {
    assert_true(params[i].grad() is None)
  }
}

///|
test "linear backward updates grad" {
  let linear = @nn.Linear::new(2, 3)
  let x = @autograd.Variable::new(@tensor.Tensor::from_array_2d([[1.0, 2.0]]))
  let output = linear.forward(x)
  let loss = output.sum()
  loss.backward()
  assert_true(linear.weight.grad() is Some(_))
  assert_true(linear.bias.grad() is Some(_))
}

///|
test "linear single sample" {
  let linear = @nn.Linear::new(3, 2)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 2.0, 3.0]]),
  )
  let output = linear.forward(x)
  assert_eq(output.shape(), [1, 2])
}

///|
test "mlp forward shape" {
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 2.0], [3.0, 4.0]]),
  )
  let h = l1.forward(x).relu()
  let output = l2.forward(h)
  assert_eq(output.shape(), [2, 1])
}

///|
test "training loss decreases" {
  // Train a single Linear to learn y = 2*x1 + 3*x2
  let linear = @nn.Linear::new(2, 1, seed=42UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
  )
  let target = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[5.0], [7.0], [8.0]]),
  )
  let lr = 0.01
  // Record initial loss
  let init_output = linear.forward(x)
  let init_diff = init_output.sub(target)
  let init_loss = init_diff.mul(init_diff).sum()
  let initial_loss_val = init_loss.data().get([])
  // Train for 50 steps
  for _step = 0; _step < 50; _step = _step + 1 {
    let output = linear.forward(x)
    let diff = output.sub(target)
    let loss = diff.mul(diff).sum()
    loss.backward()
    let params = linear.parameters()
    for i = 0; i < params.length(); i = i + 1 {
      let p = params[i]
      let grad = p.grad().unwrap()
      let updated = p.data().sub(grad.scale(lr))
      p.set_data(updated)
      p.zero_grad()
    }
  }
  // Final loss should be smaller
  let final_output = linear.forward(x)
  let final_diff = final_output.sub(target)
  let final_loss = final_diff.mul(final_diff).sum()
  let final_loss_val = final_loss.data().get([])
  assert_true(final_loss_val < initial_loss_val)
}

///|
test "training loss decreases mlp" {
  // 2-layer MLP with relu
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
  )
  let target = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[5.0], [7.0], [8.0]]),
  )
  let lr = 0.001
  // Initial loss
  let init_h = l1.forward(x).relu()
  let init_output = l2.forward(init_h)
  let init_diff = init_output.sub(target)
  let init_loss = init_diff.mul(init_diff).sum()
  let initial_loss_val = init_loss.data().get([])
  // Train for 100 steps
  for _step = 0; _step < 100; _step = _step + 1 {
    let h = l1.forward(x).relu()
    let output = l2.forward(h)
    let diff = output.sub(target)
    let loss = diff.mul(diff).sum()
    loss.backward()
    let params = l1.parameters()
    let p2 = l2.parameters()
    for i = 0; i < p2.length(); i = i + 1 {
      params.push(p2[i])
    }
    for i = 0; i < params.length(); i = i + 1 {
      let p = params[i]
      let grad = p.grad().unwrap()
      let updated = p.data().sub(grad.scale(lr))
      p.set_data(updated)
      p.zero_grad()
    }
  }
  // Final loss
  let final_h = l1.forward(x).relu()
  let final_output = l2.forward(final_h)
  let final_diff = final_output.sub(target)
  let final_loss = final_diff.mul(final_diff).sum()
  let final_loss_val = final_loss.data().get([])
  assert_true(final_loss_val < initial_loss_val)
}

///|
test "linear from_tensors creates working layer" {
  let w = @tensor.Tensor::from_array_2d([[1.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
  let b = @tensor.Tensor::from_array([0.1, 0.2, 0.3])
  let linear = @nn.Linear::from_tensors(w, b)
  let x = @autograd.Variable::new(@tensor.Tensor::from_array_2d([[2.0, 3.0]]))
  let output = linear.forward(x)
  // [2,3] @ [[1,0,-1],[0,1,2]] + [0.1,0.2,0.3] = [2.1, 3.2, 4.3]
  assert_eq(output.data().get([0, 0]), 2.1)
  assert_eq(output.data().get([0, 1]), 3.2)
  assert_eq(output.data().get([0, 2]), 4.3)
}

///|
test "mlp with tanh reduces loss" {
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
  )
  let target = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[5.0], [7.0], [8.0]]),
  )
  let lr = 0.01
  let h0 = l1.forward(x).tanh()
  let pred0 = l2.forward(h0)
  let diff0 = pred0.sub(target)
  let initial_loss = diff0.mul(diff0).sum().data().get([])
  for _step = 0; _step < 50; _step = _step + 1 {
    let h = l1.forward(x).tanh()
    let pred = l2.forward(h)
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()
    loss.backward()
    let params = l1.parameters()
    let p2 = l2.parameters()
    for i = 0; i < p2.length(); i = i + 1 {
      params.push(p2[i])
    }
    for i = 0; i < params.length(); i = i + 1 {
      let p = params[i]
      let grad = p.grad().unwrap()
      p.set_data(p.data().sub(grad.scale(lr)))
      p.zero_grad()
    }
  }
  let h_f = l1.forward(x).tanh()
  let pred_f = l2.forward(h_f)
  let diff_f = pred_f.sub(target)
  let final_loss = diff_f.mul(diff_f).sum().data().get([])
  assert_true(final_loss < initial_loss)
}

///|
test "mlp with sigmoid reduces loss" {
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
  )
  let target = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[0.5], [0.7], [0.8]]),
  )
  let lr = 0.01
  let h0 = l1.forward(x).sigmoid()
  let pred0 = l2.forward(h0)
  let diff0 = pred0.sub(target)
  let initial_loss = diff0.mul(diff0).sum().data().get([])
  for _step = 0; _step < 50; _step = _step + 1 {
    let h = l1.forward(x).sigmoid()
    let pred = l2.forward(h)
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()
    loss.backward()
    let params = l1.parameters()
    let p2 = l2.parameters()
    for i = 0; i < p2.length(); i = i + 1 {
      params.push(p2[i])
    }
    for i = 0; i < params.length(); i = i + 1 {
      let p = params[i]
      let grad = p.grad().unwrap()
      p.set_data(p.data().sub(grad.scale(lr)))
      p.zero_grad()
    }
  }
  let h_f = l1.forward(x).sigmoid()
  let pred_f = l2.forward(h_f)
  let diff_f = pred_f.sub(target)
  let final_loss = diff_f.mul(diff_f).sum().data().get([])
  assert_true(final_loss < initial_loss)
}

///|
test "self_attention forward shape" {
  let sa = @nn.SelfAttention::new(8, num_heads=2, seed=42UL)
  // input: [batch=2, seq_len=4, d_model=8]
  let input_data = @tensor.Tensor::rand([2, 4, 8], seed=1UL)
  let input = @autograd.Variable::new(input_data, requires_grad=true)
  let output = sa.forward(input)
  assert_eq(output.shape(), [2, 4, 8])
  // Verify output is finite (no NaN)
  let out_data = output.data().to_array()
  for i = 0; i < out_data.length(); i = i + 1 {
    assert_true(out_data[i] == out_data[i]) // NaN != NaN
  }
}

///|
test "self_attention single head" {
  let sa = @nn.SelfAttention::new(4, num_heads=1, seed=99UL)
  let input_data = @tensor.Tensor::rand([1, 3, 4], seed=10UL)
  let input = @autograd.Variable::new(input_data, requires_grad=true)
  let output = sa.forward(input)
  assert_eq(output.shape(), [1, 3, 4])
}

///|
test "self_attention parameters count" {
  let sa = @nn.SelfAttention::new(8, num_heads=2, seed=42UL)
  let params = sa.parameters()
  // 4 Linear layers × 2 params each (weight + bias) = 8
  assert_eq(params.length(), 8)
}

///|
test "self_attention backward grad exists" {
  let sa = @nn.SelfAttention::new(4, num_heads=1, seed=42UL)
  let input_data = @tensor.Tensor::rand([1, 2, 4], seed=1UL)
  let input = @autograd.Variable::new(input_data, requires_grad=true)
  let output = sa.forward(input)
  let loss = output.sum()
  loss.backward()
  assert_true(input.grad() is Some(_))
}

///|
test "self_attention training loss decreases" {
  let sa = @nn.SelfAttention::new(4, num_heads=1, seed=42UL)
  let input_data = @tensor.Tensor::rand([1, 2, 4], seed=1UL)
  let target_data = @tensor.Tensor::rand([1, 2, 4], seed=2UL)
  let input = @autograd.Variable::new(input_data)
  let target = @autograd.Variable::new(target_data)
  let lr = 0.01
  // Initial loss
  let out0 = sa.forward(input)
  let diff0 = out0.sub(target)
  let initial_loss = diff0.mul(diff0).sum().data().get([])
  // Train for 5 steps
  for _step = 0; _step < 5; _step = _step + 1 {
    let out = sa.forward(input)
    let diff = out.sub(target)
    let loss = diff.mul(diff).sum()
    loss.backward()
    let params = sa.parameters()
    for i = 0; i < params.length(); i = i + 1 {
      let p = params[i]
      let grad = p.grad().unwrap()
      p.set_data(p.data().sub(grad.scale(lr)))
      p.zero_grad()
    }
  }
  let out_f = sa.forward(input)
  let diff_f = out_f.sub(target)
  let final_loss = diff_f.mul(diff_f).sum().data().get([])
  assert_true(final_loss < initial_loss)
}
