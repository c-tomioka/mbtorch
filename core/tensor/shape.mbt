///|
pub fn Tensor::shape(self : Tensor) -> Array[Int] {
  self.shape_.copy()
}

///|
pub fn Tensor::ndim(self : Tensor) -> Int {
  self.shape_.length()
}

///|
pub fn Tensor::size(self : Tensor) -> Int {
  self.data_.length()
}

///|
pub fn Tensor::get(self : Tensor, index : Array[Int]) -> Double {
  let offset = flat_index(self.shape_, index)
  self.data_[offset]
}

///|
fn flat_index(shape : Array[Int], index : Array[Int]) -> Int {
  guard index.length() == shape.length() else {
    abort(
      "index dimensions \{index.length()} does not match tensor dimensions \{shape.length()}",
    )
  }
  let mut offset = 0
  for i = 0; i < shape.length(); i = i + 1 {
    offset = offset * shape[i] + index[i]
  }
  offset
}
