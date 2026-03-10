///|
pub fn Tensor::matmul(self : Tensor, other : Tensor) -> Tensor {
  guard self.ndim() == 2 && other.ndim() == 2 else {
    abort("matmul: both tensors must be 2D")
  }
  let m = self.shape_[0]
  let k = self.shape_[1]
  let k2 = other.shape_[0]
  let n = other.shape_[1]
  guard k == k2 else {
    abort("matmul: inner dimensions mismatch \{k} vs \{k2}")
  }
  let result = Array::make(m * n, 0.0)
  for i = 0; i < m; i = i + 1 {
    for j = 0; j < n; j = j + 1 {
      let mut sum = 0.0
      for p = 0; p < k; p = p + 1 {
        sum = sum + self.data_[i * k + p] * other.data_[p * n + j]
      }
      result[i * n + j] = sum
    }
  }
  { data_: result, shape_: [m, n], dtype_: Float32, quant_params_: None }
}

///|
pub fn Tensor::transpose(self : Tensor) -> Tensor {
  guard self.ndim() == 2 else { abort("transpose: tensor must be 2D") }
  let rows = self.shape_[0]
  let cols = self.shape_[1]
  let result = Array::make(rows * cols, 0.0)
  for i = 0; i < rows; i = i + 1 {
    for j = 0; j < cols; j = j + 1 {
      result[j * rows + i] = self.data_[i * cols + j]
    }
  }
  { data_: result, shape_: [cols, rows], dtype_: Float32, quant_params_: None }
}
