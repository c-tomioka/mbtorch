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

///|
pub fn Tensor::sub(self : Tensor, other : Tensor) -> Tensor {
  guard self.shape_ == other.shape_ else {
    abort("sub: shape mismatch \{self.shape_} vs \{other.shape_}")
  }
  let n = self.data_.length()
  let result = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    result[i] = self.data_[i] - other.data_[i]
  }
  { data_: result, shape_: self.shape_.copy() }
}

///|
pub fn Tensor::scale(self : Tensor, factor : Double) -> Tensor {
  let n = self.data_.length()
  let result = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    result[i] = self.data_[i] * factor
  }
  { data_: result, shape_: self.shape_.copy() }
}

///|
pub fn Tensor::sum(self : Tensor) -> Tensor {
  let mut total = 0.0
  for i = 0; i < self.data_.length(); i = i + 1 {
    total = total + self.data_[i]
  }
  { data_: [total], shape_: [] }
}

///|
/// Element-wise division.
pub fn Tensor::div(self : Tensor, other : Tensor) -> Tensor {
  guard self.shape_ == other.shape_ else {
    abort("div: shape mismatch \{self.shape_} vs \{other.shape_}")
  }
  let n = self.data_.length()
  let result = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    result[i] = self.data_[i] / other.data_[i]
  }
  { data_: result, shape_: self.shape_.copy() }
}

///|
/// Element-wise square root.
pub fn Tensor::sqrt(self : Tensor) -> Tensor {
  let n = self.data_.length()
  let result = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    result[i] = self.data_[i].sqrt()
  }
  { data_: result, shape_: self.shape_.copy() }
}
