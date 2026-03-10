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
    Some(TanhBackward(a)) => {
      // d/dx tanh(x) = 1 - tanh(x)^2
      let n = a.data.size()
      let shape = a.data.shape()
      let result = Array::make(n, 0.0)
      for i = 0; i < n; i = i + 1 {
        let tanh_val = tanh_scalar(get_flat_val(a.data, i))
        let grad_val = get_flat_val(grad, i)
        result[i] = grad_val * (1.0 - tanh_val * tanh_val)
      }
      backward_with_grad(a, rebuild_tensor(result, shape))
    }
    Some(SigmoidBackward(a)) => {
      // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
      let n = a.data.size()
      let shape = a.data.shape()
      let result = Array::make(n, 0.0)
      for i = 0; i < n; i = i + 1 {
        let sig_val = sigmoid_scalar(get_flat_val(a.data, i))
        let grad_val = get_flat_val(grad, i)
        result[i] = grad_val * sig_val * (1.0 - sig_val)
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
    Some(ScaleBackward(a, factor)) => {
      backward_with_grad(a, grad.scale(factor))
    }
    Some(SoftmaxBackward(input, softmax_data)) => {
      // grad_input = softmax_out * (grad - sum(grad * softmax_out, per row))
      let s = softmax_data.to_array()
      let g = grad.to_array()
      let n = s.length()
      let shape = softmax_data.shape()
      let last_dim = shape[shape.length() - 1]
      let num_rows = n / last_dim
      let result = Array::make(n, 0.0)
      for row = 0; row < num_rows; row = row + 1 {
        let offset = row * last_dim
        let mut dot = 0.0
        for j = 0; j < last_dim; j = j + 1 {
          dot = dot + g[offset + j] * s[offset + j]
        }
        for j = 0; j < last_dim; j = j + 1 {
          result[offset + j] = s[offset + j] * (g[offset + j] - dot)
        }
      }
      backward_with_grad(
        input,
        @tensor.Tensor::from_array_with_shape(result, shape),
      )
    }
    Some(BatchedMatmulBackward(a, b)) => {
      // d/da = grad @ b^T_last2, d/db = a^T_last2 @ grad
      backward_with_grad(a, grad.batched_matmul_2d(b.data.transpose_last2()))
      backward_with_grad(b, a.data.transpose_last2().batched_matmul_2d(grad))
    }
    Some(TransposeLast2Backward(a)) => {
      // transpose_last2 is its own inverse
      backward_with_grad(a, grad.transpose_last2())
    }
    Some(ReshapeBackward(a, original_shape)) => {
      backward_with_grad(a, grad.reshape(original_shape))
    }
    None => ()
  }
}
