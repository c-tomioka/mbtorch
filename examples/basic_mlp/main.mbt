///|
/// 2-layer MLP training demo
/// Model: Linear(2,4) -> ReLU -> Linear(4,1)
/// Data:  y = 2*x1 + 3*x2 (synthetic)
fn main {
  // Model
  let l1 = @nn.Linear::new(2, 4, seed=42UL)
  let l2 = @nn.Linear::new(4, 1, seed=123UL)

  // Synthetic data: y = 2*x1 + 3*x2
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
  )
  let target = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[5.0], [7.0], [8.0]]),
  )

  // Optimizer
  let params = l1.parameters()
  params.append(l2.parameters())
  let optimizer = @optim.SGD::new(params, lr=0.001)

  for step = 0; step < 100; step = step + 1 {
    // Forward
    let h = l1.forward(x).relu()
    let pred = l2.forward(h)

    // MSE loss
    let diff = pred.sub(target)
    let loss = diff.mul(diff).sum()

    // Log every 10 steps
    if step % 10 == 0 {
      println("step \{step}: loss = \{loss.data()}")
    }

    // Backward + optimize
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
  }
  println("Training complete.")
}
