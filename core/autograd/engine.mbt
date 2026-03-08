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
    Some(SubBackward(a, b)) => {
      // d/da(a-b) = 1, d/db(a-b) = -1
      backward_with_grad(a, grad)
      let neg_grad = grad.scale(-1.0)
      backward_with_grad(b, neg_grad)
    }
    Some(SumBackward(a)) => {
      let grad_val = grad.get([])
      let expanded = @tensor.Tensor::fill(a.data.shape(), grad_val)
      backward_with_grad(a, expanded)
    }
    Some(ReluBackward(a)) => {
      // grad * (input > 0 ? 1 : 0)
      let n = a.data.size()
      let shape = a.data.shape()
      let result = Array::make(n, 0.0)
      for i = 0; i < n; i = i + 1 {
        let input_val = get_flat_val(a.data, i)
        let grad_val = get_flat_val(grad, i)
        result[i] = if input_val > 0.0 { grad_val } else { 0.0 }
      }
      backward_with_grad(a, rebuild_tensor(result, shape))
    }
    Some(ExpandRowsBackward(a)) => {
      // Sum gradient along batch dimension: [batch, cols] -> [cols]
      let cols = a.data.shape()[0]
      let batch = grad.shape()[0]
      let summed = Array::make(cols, 0.0)
      for r = 0; r < batch; r = r + 1 {
        for c = 0; c < cols; c = c + 1 {
          summed[c] = summed[c] + grad.get([r, c])
        }
      }
      backward_with_grad(a, @tensor.Tensor::from_array(summed))
    }
    None => ()
  }
}
