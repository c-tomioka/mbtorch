///|
/// Convert a Double value to IEEE 754 float16 little-endian (2 bytes).
fn double_to_float16_le(value : Double) -> Array[Byte] {
  let bits = @tensor.double_to_float16_bits(value)
  let lo = (bits & 0xFF).to_byte()
  let hi = ((bits >> 8) & 0xFF).to_byte()
  [lo, hi]
}

///|
/// Read 2 little-endian bytes as IEEE 754 float16 and return a Double.
fn float16_le_to_double(data : Bytes, offset : Int) -> Double {
  let lo = data[offset].to_int()
  let hi = data[offset + 1].to_int()
  let bits = lo | (hi << 8)
  @tensor.float16_bits_to_double(bits)
}

///|
/// Quantize a Double value to uint8 (0..255) given scale and zero_point.
fn double_to_uint8(value : Double, scale : Double, zero_point : Int) -> Byte {
  let mut q = (value / scale + zero_point.to_double() + 0.5).to_int()
  if q < 0 { q = 0 }
  if q > 255 { q = 255 }
  q.to_byte()
}

///|
/// Dequantize a uint8 byte to Double given scale and zero_point.
fn uint8_to_double(b : Byte, scale : Double, zero_point : Int) -> Double {
  (b.to_int() - zero_point).to_double() * scale
}
