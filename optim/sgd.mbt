///|
/// Stochastic Gradient Descent optimizer.
/// Updates parameters using: param = param - lr * grad
pub struct SGD {
  params : Array[@autograd.Variable]
  lr : Double
}

///|
pub fn SGD::new(params : Array[@autograd.Variable], lr? : Double = 0.01) -> SGD {
  { params, lr }
}

///|
/// Apply one step of SGD: param.data -= lr * param.grad for all params.
/// Skips parameters whose grad is None.
pub fn SGD::step(self : SGD) -> Unit {
  for i = 0; i < self.params.length(); i = i + 1 {
    let p = self.params[i]
    match p.grad() {
      Some(grad) => p.set_data(p.data().sub(grad.scale(self.lr)))
      None => ()
    }
  }
}

///|
/// Zero all parameter gradients.
pub fn SGD::zero_grad(self : SGD) -> Unit {
  for i = 0; i < self.params.length(); i = i + 1 {
    self.params[i].zero_grad()
  }
}
