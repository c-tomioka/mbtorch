///|
/// Convert a Double to IEEE 754 float16 bit representation (as Int, 0..65535).
/// sign: 1 bit, exponent: 5 bits (bias=15), mantissa: 10 bits
pub fn double_to_float16_bits(value : Double) -> Int {
  let abs_val = value.abs()
  let sign = if value < 0.0 { 1 } else { 0 }
  // Zero
  if abs_val == 0.0 {
    return (sign << 15)
  }
  // Infinity
  if abs_val > 1.0e30 {
    return (sign << 15) | 0x7C00
  }
  // NaN check: NaN != NaN
  if value != value {
    return 0x7E00 // quiet NaN
  }
  // Overflow: f16 max is 65504
  if abs_val > 65504.0 {
    return (sign << 15) | 0x7C00
  }
  // Underflow: f16 min subnormal is 2^-24 ≈ 5.96e-8
  if abs_val < 5.96e-8 {
    return (sign << 15)
  }
  // Subnormal range: abs_val < 2^-14 ≈ 6.1035e-5
  if abs_val < 6.103515625e-5 {
    let mantissa = (abs_val * 16777216.0 + 0.5).to_int() // abs_val * 2^24
    let m = if mantissa > 0x3FF { 0x3FF } else { mantissa }
    return (sign << 15) | m
  }
  // Normal range: compute exponent and mantissa
  let mut exp = 15
  let mut frac = abs_val
  if frac >= 2.0 {
    while frac >= 2.0 && exp < 30 {
      frac = frac / 2.0
      exp = exp + 1
    }
  } else {
    while frac < 1.0 && exp > 1 {
      frac = frac * 2.0
      exp = exp - 1
    }
  }
  // frac is now in [1.0, 2.0), mantissa = (frac - 1.0) * 1024
  let mantissa = ((frac - 1.0) * 1024.0 + 0.5).to_int()
  let m = if mantissa > 0x3FF { 0x3FF } else { mantissa }
  (sign << 15) | (exp << 10) | m
}

///|
/// Convert IEEE 754 float16 bit representation (as Int) back to Double.
pub fn float16_bits_to_double(bits : Int) -> Double {
  let b = bits.reinterpret_as_uint()
  let sign = (b >> 15) & 1U
  let exp = ((b >> 10) & 0x1FU).reinterpret_as_int()
  let mantissa = (b & 0x3FFU).reinterpret_as_int()
  let sign_mul = if sign == 1U { -1.0 } else { 1.0 }
  // Zero
  if exp == 0 && mantissa == 0 {
    return sign_mul * 0.0
  }
  // Subnormal
  if exp == 0 {
    return sign_mul * mantissa.to_double() * 5.9604644775390625e-8 // 2^-24
  }
  // Infinity
  if exp == 31 && mantissa == 0 {
    return sign_mul * (1.0 / 0.0)
  }
  // NaN
  if exp == 31 {
    return 0.0 / 0.0
  }
  // Normal: value = (-1)^sign * 2^(exp-15) * (1 + mantissa/1024)
  let mut power = 1.0
  let e = exp - 15
  if e >= 0 {
    for i = 0; i < e; i = i + 1 {
      power = power * 2.0
    }
  } else {
    for i = 0; i < -e; i = i + 1 {
      power = power / 2.0
    }
  }
  sign_mul * power * (1.0 + mantissa.to_double() / 1024.0)
}

///|
/// Quantize an array of Double values to int8 using asymmetric linear quantization.
/// Returns (quantized_values, scale, zero_point).
/// Maps [min_val, max_val] to [0, 255].
pub fn quantize_to_int8(
  data : Array[Double],
) -> (Array[Int], Double, Int) {
  guard data.length() > 0 else { return ([], 1.0, 0) }
  let mut min_val = data[0]
  let mut max_val = data[0]
  for i = 1; i < data.length(); i = i + 1 {
    if data[i] < min_val { min_val = data[i] }
    if data[i] > max_val { max_val = data[i] }
  }
  let range = max_val - min_val
  let scale = if range < 1.0e-10 { 1.0 } else { range / 255.0 }
  let zero_point_f = -min_val / scale
  let mut zp = (zero_point_f + 0.5).to_int()
  if zp < 0 { zp = 0 }
  if zp > 255 { zp = 255 }
  let quantized = Array::make(data.length(), 0)
  for i = 0; i < data.length(); i = i + 1 {
    let mut q = (data[i] / scale + zp.to_double() + 0.5).to_int()
    if q < 0 { q = 0 }
    if q > 255 { q = 255 }
    quantized[i] = q
  }
  (quantized, scale, zp)
}

///|
/// Dequantize int8 values back to Double.
/// real_value = (quantized_value - zero_point) * scale
pub fn dequantize_from_int8(
  data : Array[Int],
  scale : Double,
  zero_point : Int,
) -> Array[Double] {
  let result = Array::make(data.length(), 0.0)
  for i = 0; i < data.length(); i = i + 1 {
    result[i] = (data[i] - zero_point).to_double() * scale
  }
  result
}

///|
/// Convert a tensor to a different storage dtype.
/// For Float16/Float32/Float64: only changes the dtype tag, data stays as Double.
/// For Int8: computes quantization parameters from the data range.
pub fn Tensor::to_dtype(self : Tensor, target : DType) -> Tensor {
  match target {
    Int8 => {
      let (_, scale, zp) = quantize_to_int8(self.data_)
      {
        data_: self.data_.copy(),
        shape_: self.shape_.copy(),
        dtype_: Int8,
        quant_params_: Some(QuantParams::{ scale, zero_point: zp }),
      }
    }
    Float16 | Float32 | Float64 =>
      {
        data_: self.data_.copy(),
        shape_: self.shape_.copy(),
        dtype_: target,
        quant_params_: None,
      }
  }
}
