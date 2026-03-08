///|
/// Adam optimizer (Kingma & Ba, 2014).
/// Adaptive learning rate using first and second moment estimates.
pub struct Adam {
  params : Array[@autograd.Variable]
  lr : Double
  beta1 : Double
  beta2 : Double
  epsilon : Double
  mut t : Int
  m : Array[@tensor.Tensor]
  v : Array[@tensor.Tensor]
}

///|
pub fn Adam::new(
  params : Array[@autograd.Variable],
  lr? : Double = 0.001,
  beta1? : Double = 0.9,
  beta2? : Double = 0.999,
  epsilon? : Double = 1.0e-8,
) -> Adam {
  let m : Array[@tensor.Tensor] = []
  let v : Array[@tensor.Tensor] = []
  for i = 0; i < params.length(); i = i + 1 {
    let shape = params[i].shape()
    m.push(@tensor.Tensor::zeros(shape))
    v.push(@tensor.Tensor::zeros(shape))
  }
  { params, lr, beta1, beta2, epsilon, t: 0, m, v }
}

///|
/// Apply one step of Adam update.
/// Skips parameters whose grad is None.
pub fn Adam::step(self : Adam) -> Unit {
  self.t = self.t + 1
  for i = 0; i < self.params.length(); i = i + 1 {
    let p = self.params[i]
    match p.grad() {
      Some(grad) => {
        // m[i] = beta1 * m[i] + (1 - beta1) * grad
        self.m[i] = self.m[i]
          .scale(self.beta1)
          .add(grad.scale(1.0 - self.beta1))
        // v[i] = beta2 * v[i] + (1 - beta2) * grad^2
        self.v[i] = self.v[i]
          .scale(self.beta2)
          .add(grad.mul(grad).scale(1.0 - self.beta2))
        // Bias correction
        let m_hat = self.m[i].scale(1.0 / (1.0 - pow(self.beta1, self.t)))
        let v_hat = self.v[i].scale(1.0 / (1.0 - pow(self.beta2, self.t)))
        // param -= lr * m_hat / (sqrt(v_hat) + epsilon)
        let denom = v_hat
          .sqrt()
          .add(@tensor.Tensor::fill(v_hat.shape(), self.epsilon))
        let update = m_hat.div(denom).scale(self.lr)
        p.set_data(p.data().sub(update))
      }
      None => ()
    }
  }
}

///|
/// Zero all parameter gradients.
pub fn Adam::zero_grad(self : Adam) -> Unit {
  for i = 0; i < self.params.length(); i = i + 1 {
    self.params[i].zero_grad()
  }
}

///|
/// Compute base^exp for positive integer exponents.
fn pow(base : Double, exp : Int) -> Double {
  let mut result = 1.0
  for _i = 0; _i < exp; _i = _i + 1 {
    result = result * base
  }
  result
}
