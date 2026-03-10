///|
/// Softmax along the last axis.
/// For shape [..., N], computes softmax independently over each row of N elements.
/// Uses max-subtract for numerical stability.
pub fn Tensor::softmax(self : Tensor) -> Tensor {
  guard self.ndim() >= 1 else {
    abort("softmax: tensor must be at least 1D")
  }
  let shape = self.shape_
  let last_dim = shape[shape.length() - 1]
  let total = self.data_.length()
  let num_rows = total / last_dim
  let result = Array::make(total, 0.0)
  for row = 0; row < num_rows; row = row + 1 {
    let offset = row * last_dim
    // Find max for numerical stability
    let mut max_val = self.data_[offset]
    for j = 1; j < last_dim; j = j + 1 {
      if self.data_[offset + j] > max_val {
        max_val = self.data_[offset + j]
      }
    }
    // Compute exp(x - max) and sum
    let mut sum_exp = 0.0
    for j = 0; j < last_dim; j = j + 1 {
      let e = @math.exp(self.data_[offset + j] - max_val)
      result[offset + j] = e
      sum_exp = sum_exp + e
    }
    // Normalize
    for j = 0; j < last_dim; j = j + 1 {
      result[offset + j] = result[offset + j] / sum_exp
    }
  }
  { data_: result, shape_: shape.copy(), dtype_: Float32, quant_params_: None }
}
