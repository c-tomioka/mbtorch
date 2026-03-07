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
  { data_: result, shape_: [m, n] }
}
