///|
/// Tensor represents a multi-dimensional array of Double values.
/// Data is stored in row-major (C-contiguous) order.
struct Tensor {
  data_ : Array[Double]
  shape_ : Array[Int]
}
