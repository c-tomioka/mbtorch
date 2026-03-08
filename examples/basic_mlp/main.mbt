///|
/// 2-layer MLP training demo: SGD vs Adam comparison
/// Model: Linear(2,4) -> ReLU -> Linear(4,1)
/// Data:  y = 2*x1 + 3*x2 (synthetic)
fn main {
  // Synthetic data: y = 2*x1 + 3*x2
  let x_data = @tensor.Tensor::from_array_2d([
    [1.0, 1.0],
    [2.0, 1.0],
    [1.0, 2.0],
  ])
  let t_data = @tensor.Tensor::from_array_2d([[5.0], [7.0], [8.0]])

  // --- SGD ---
  println("=== SGD (lr=0.001) ===")
  let l1_sgd = @nn.Linear::new(2, 4, seed=42UL)
  let l2_sgd = @nn.Linear::new(4, 1, seed=123UL)
  let x_sgd = @autograd.Variable::new(x_data)
  let target_sgd = @autograd.Variable::new(t_data)
  let params_sgd = l1_sgd.parameters()
  params_sgd.append(l2_sgd.parameters())
  let sgd = @optim.SGD::new(params_sgd, lr=0.001)
  for step = 0; step < 100; step = step + 1 {
    let h = l1_sgd.forward(x_sgd).relu()
    let pred = l2_sgd.forward(h)
    let diff = pred.sub(target_sgd)
    let loss = diff.mul(diff).sum()
    if step % 10 == 0 {
      println("step \{step}: loss = \{loss.data()}")
    }
    loss.backward()
    sgd.step()
    sgd.zero_grad()
  }

  // --- Adam ---
  println("=== Adam (lr=0.05) ===")
  let l1_adam = @nn.Linear::new(2, 4, seed=42UL)
  let l2_adam = @nn.Linear::new(4, 1, seed=123UL)
  let x_adam = @autograd.Variable::new(x_data)
  let target_adam = @autograd.Variable::new(t_data)
  let params_adam = l1_adam.parameters()
  params_adam.append(l2_adam.parameters())
  let adam = @optim.Adam::new(params_adam, lr=0.05)
  for step = 0; step < 100; step = step + 1 {
    let h = l1_adam.forward(x_adam).relu()
    let pred = l2_adam.forward(h)
    let diff = pred.sub(target_adam)
    let loss = diff.mul(diff).sum()
    if step % 10 == 0 {
      println("step \{step}: loss = \{loss.data()}")
    }
    loss.backward()
    adam.step()
    adam.zero_grad()
  }

  println("Training complete.")
}
