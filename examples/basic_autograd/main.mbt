///|
fn main {
  // 1. Create Variables with requires_grad=true
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array([1.0, 2.0, 3.0]),
    requires_grad=true,
  )
  let w = @autograd.Variable::new(
    @tensor.Tensor::from_array([0.5, -1.0, 2.0]),
    requires_grad=true,
  )
  let b = @autograd.Variable::new(
    @tensor.Tensor::from_array([0.1, 0.2, 0.3]),
    requires_grad=true,
  )

  // 2. Forward: loss = sum(x * w + b)
  let product = x.mul(w)
  let biased = product.add(b)
  let loss = biased.sum()

  println("=== Forward ===")
  println("x:    \{x.data()}")
  println("w:    \{w.data()}")
  println("b:    \{b.data()}")
  println("loss: \{loss.data()}")

  // 3. Backward: compute gradients
  loss.backward()

  println("\n=== Backward (gradients) ===")
  println("x.grad: \{x.grad().unwrap()}")
  println("w.grad: \{w.grad().unwrap()}")
  println("b.grad: \{b.grad().unwrap()}")
}
