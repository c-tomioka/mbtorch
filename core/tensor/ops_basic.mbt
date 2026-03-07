///|
pub fn Tensor::add(self : Tensor, other : Tensor) -> Tensor {
  guard self.shape_ == other.shape_ else {
    abort("add: shape mismatch \{self.shape_} vs \{other.shape_}")
  }
  let n = self.data_.length()
  let result = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    result[i] = self.data_[i] + other.data_[i]
  }
  { data_: result, shape_: self.shape_.copy() }
}

///|
pub fn Tensor::mul(self : Tensor, other : Tensor) -> Tensor {
  guard self.shape_ == other.shape_ else {
    abort("mul: shape mismatch \{self.shape_} vs \{other.shape_}")
  }
  let n = self.data_.length()
  let result = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    result[i] = self.data_[i] * other.data_[i]
  }
  { data_: result, shape_: self.shape_.copy() }
}

///|
pub impl Add for Tensor with add(self, other) {
  self.add(other)
}

///|
pub impl Mul for Tensor with mul(self, other) {
  self.mul(other)
}
