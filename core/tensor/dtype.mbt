///|
/// Data type for tensor storage encoding.
/// All computation is performed in Double; dtype controls IO serialization only.
pub(all) enum DType {
  Float64
  Float32
  Float16
  Int8
} derive(Eq, Show)

///|
/// Quantization parameters for Int8 linear quantization.
/// real_value = (int8_value - zero_point) * scale
pub(all) struct QuantParams {
  scale : Double
  zero_point : Int
} derive(Eq, Show)
