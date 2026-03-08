///|
pub fn Variable::backward(self : Variable) -> Unit {
  // Initialize output gradient to ones with same shape
  let ones = @tensor.Tensor::ones(self.data.shape())
  backward_with_grad(self, ones)
}

///|
fn backward_with_grad(variable : Variable, grad : @tensor.Tensor) -> Unit {
  // Accumulate gradient if this variable requires grad
  if variable.requires_grad {
    match variable.grad {
      None => variable.grad = Some(grad)
      Some(existing) => variable.grad = Some(existing.add(grad))
    }
  }
  // Propagate to parents via grad_fn
  match variable.grad_fn {
    Some(AddBackward(a, b)) => {
      backward_with_grad(a, grad)
      backward_with_grad(b, grad)
    }
    Some(MulBackward(a, b)) => {
      // d/da(a*b) = b, d/db(a*b) = a
      backward_with_grad(a, grad.mul(b.data))
      backward_with_grad(b, grad.mul(a.data))
    }
    Some(MatmulBackward(a, b)) => {
      // d/da(a@b) = grad @ b^T, d/db(a@b) = a^T @ grad
      backward_with_grad(a, grad.matmul(b.data.transpose()))
      backward_with_grad(b, a.data.transpose().matmul(grad))
    }
    Some(SumBackward(a)) => {
      let grad_val = grad.get([])
      let expanded = @tensor.Tensor::fill(a.data.shape(), grad_val)
      backward_with_grad(a, expanded)
    }
    None => ()
  }
}
