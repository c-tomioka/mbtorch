///|
/// Tensor represents a multi-dimensional array of Double values.
/// Data is stored in row-major (C-contiguous) order.
/// dtype_ controls IO serialization encoding; computation is always in Double.
struct Tensor {
  data_ : Array[Double]
  shape_ : Array[Int]
  dtype_ : DType
  quant_params_ : QuantParams?
}
