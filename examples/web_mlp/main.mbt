///|
/// Browser-based 2-layer MLP training demo (wasm-gc)
/// Compares 6 optimizer+activation combinations:
///   SGD  × {ReLU, tanh, sigmoid}
///   Adam × {ReLU, tanh, sigmoid}
/// Model: Linear(2,4) -> activation -> Linear(4,1)
/// Data:  y = 2*x1 + 3*x2 (synthetic, 3 samples)
fn main {
  let x_data = @tensor.Tensor::from_array_2d([
    [1.0, 1.0],
    [2.0, 1.0],
    [1.0, 2.0],
  ])
  let t_data = @tensor.Tensor::from_array_2d([[5.0], [7.0], [8.0]])

  // --- 1. SGD + ReLU ---
  println("=== SGD + ReLU ===")
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(x_data)
  let target = @autograd.Variable::new(t_data)
  let params = l1.parameters()
  params.append(l2.parameters())
  let opt = @optim.SGD::new(params, lr=0.001)
  for step = 0; step < 100; step = step + 1 {
    let h = l1.forward(x).relu()
    let pred = l2.forward(h)
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()
    let loss_val = loss.data().get([])
    println("DATA:SGD + ReLU,\{step},\{loss_val}")
    if step % 10 == 0 {
      println("step \{step}: loss = \{loss_val}")
    }
    loss.backward()
    opt.step()
    opt.zero_grad()
  }
  println("Training complete.")

  // --- 2. SGD + tanh ---
  println("=== SGD + tanh ===")
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(x_data)
  let target = @autograd.Variable::new(t_data)
  let params = l1.parameters()
  params.append(l2.parameters())
  let opt = @optim.SGD::new(params, lr=0.001)
  for step = 0; step < 100; step = step + 1 {
    let h = l1.forward(x).tanh()
    let pred = l2.forward(h)
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()
    let loss_val = loss.data().get([])
    println("DATA:SGD + tanh,\{step},\{loss_val}")
    if step % 10 == 0 {
      println("step \{step}: loss = \{loss_val}")
    }
    loss.backward()
    opt.step()
    opt.zero_grad()
  }
  println("Training complete.")

  // --- 3. SGD + sigmoid ---
  println("=== SGD + sigmoid ===")
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(x_data)
  let target = @autograd.Variable::new(t_data)
  let params = l1.parameters()
  params.append(l2.parameters())
  let opt = @optim.SGD::new(params, lr=0.001)
  for step = 0; step < 100; step = step + 1 {
    let h = l1.forward(x).sigmoid()
    let pred = l2.forward(h)
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()
    let loss_val = loss.data().get([])
    println("DATA:SGD + sigmoid,\{step},\{loss_val}")
    if step % 10 == 0 {
      println("step \{step}: loss = \{loss_val}")
    }
    loss.backward()
    opt.step()
    opt.zero_grad()
  }
  println("Training complete.")

  // --- 4. Adam + ReLU ---
  println("=== Adam + ReLU ===")
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(x_data)
  let target = @autograd.Variable::new(t_data)
  let params = l1.parameters()
  params.append(l2.parameters())
  let opt = @optim.Adam::new(params, lr=0.05)
  for step = 0; step < 100; step = step + 1 {
    let h = l1.forward(x).relu()
    let pred = l2.forward(h)
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()
    let loss_val = loss.data().get([])
    println("DATA:Adam + ReLU,\{step},\{loss_val}")
    if step % 10 == 0 {
      println("step \{step}: loss = \{loss_val}")
    }
    loss.backward()
    opt.step()
    opt.zero_grad()
  }
  println("Training complete.")

  // --- 5. Adam + tanh ---
  println("=== Adam + tanh ===")
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(x_data)
  let target = @autograd.Variable::new(t_data)
  let params = l1.parameters()
  params.append(l2.parameters())
  let opt = @optim.Adam::new(params, lr=0.05)
  for step = 0; step < 100; step = step + 1 {
    let h = l1.forward(x).tanh()
    let pred = l2.forward(h)
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()
    let loss_val = loss.data().get([])
    println("DATA:Adam + tanh,\{step},\{loss_val}")
    if step % 10 == 0 {
      println("step \{step}: loss = \{loss_val}")
    }
    loss.backward()
    opt.step()
    opt.zero_grad()
  }
  println("Training complete.")

  // --- 6. Adam + sigmoid ---
  println("=== Adam + sigmoid ===")
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)
  let x = @autograd.Variable::new(x_data)
  let target = @autograd.Variable::new(t_data)
  let params = l1.parameters()
  params.append(l2.parameters())
  let opt = @optim.Adam::new(params, lr=0.05)
  for step = 0; step < 100; step = step + 1 {
    let h = l1.forward(x).sigmoid()
    let pred = l2.forward(h)
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()
    let loss_val = loss.data().get([])
    println("DATA:Adam + sigmoid,\{step},\{loss_val}")
    if step % 10 == 0 {
      println("step \{step}: loss = \{loss_val}")
    }
    loss.backward()
    opt.step()
    opt.zero_grad()
  }
  println("Training complete.")
}
