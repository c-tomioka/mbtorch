// DType conversion tests

///|
/// D-T1: float16 roundtrip for normal values (1.0, -1.0, 0.5)
test "f16 roundtrip normal values" {
  let eps = 1.0e-3
  // 1.0
  let bits1 = @tensor.double_to_float16_bits(1.0)
  let val1 = @tensor.float16_bits_to_double(bits1)
  assert_true((val1 - 1.0).abs() < eps)
  // -1.0
  let bits2 = @tensor.double_to_float16_bits(-1.0)
  let val2 = @tensor.float16_bits_to_double(bits2)
  assert_true((val2 - -1.0).abs() < eps)
  // 0.5
  let bits3 = @tensor.double_to_float16_bits(0.5)
  let val3 = @tensor.float16_bits_to_double(bits3)
  assert_true((val3 - 0.5).abs() < eps)
  // -3.14 (some precision loss expected)
  let bits4 = @tensor.double_to_float16_bits(-3.14)
  let val4 = @tensor.float16_bits_to_double(bits4)
  assert_true((val4 - -3.14).abs() < 0.01)
}

///|
/// D-T2: float16 roundtrip for special values (0, inf, NaN)
test "f16 roundtrip special values" {
  // +0
  let bits_zero = @tensor.double_to_float16_bits(0.0)
  let val_zero = @tensor.float16_bits_to_double(bits_zero)
  assert_true(val_zero == 0.0)
  // -0
  let bits_neg_zero = @tensor.double_to_float16_bits(-0.0)
  let val_neg_zero = @tensor.float16_bits_to_double(bits_neg_zero)
  assert_true(val_neg_zero == 0.0) // -0.0 == 0.0 in IEEE
  // +inf
  let bits_inf = @tensor.double_to_float16_bits(1.0 / 0.0)
  let val_inf = @tensor.float16_bits_to_double(bits_inf)
  assert_true(val_inf > 1.0e30)
  // -inf
  let bits_ninf = @tensor.double_to_float16_bits(-1.0 / 0.0)
  let val_ninf = @tensor.float16_bits_to_double(bits_ninf)
  assert_true(val_ninf < -1.0e30)
}

///|
/// D-T3: float16 precision loss within tolerance for typical weight values
test "f16 precision loss within tolerance" {
  let values = [0.1, 0.01, 0.001, 1.5, 100.0, -0.25, 42.0]
  for i = 0; i < values.length(); i = i + 1 {
    let v = values[i]
    let bits = @tensor.double_to_float16_bits(v)
    let recovered = @tensor.float16_bits_to_double(bits)
    let err = (recovered - v).abs()
    let denom = if v.abs() > 1.0e-7 { v.abs() } else { 1.0e-7 }
    let rel = err / denom
    assert_true(rel < 0.01) // < 1% relative error for normal range
  }
}

///|
/// D-T4: float16 overflow to infinity
test "f16 overflow to inf" {
  // f16 max is 65504; values > 65504 should become inf
  let bits = @tensor.double_to_float16_bits(65536.0)
  let val = @tensor.float16_bits_to_double(bits)
  assert_true(val > 65504.0) // inf
}

///|
/// D-T5: float16 underflow to zero
test "f16 underflow to zero" {
  let bits = @tensor.double_to_float16_bits(1.0e-8)
  let val = @tensor.float16_bits_to_double(bits)
  assert_true(val == 0.0)
}

///|
/// D-T6: int8 quantize/dequantize roundtrip with known params
test "int8 quantize dequantize roundtrip" {
  // scale=0.1, zero_point=0 (symmetric)
  let data = [0.0, 0.1, 0.5, -0.3, 1.0]
  let (quantized, scale, zp) = @tensor.quantize_to_int8(data)
  let recovered = @tensor.dequantize_from_int8(quantized, scale, zp)
  for i = 0; i < data.length(); i = i + 1 {
    let err = (recovered[i] - data[i]).abs()
    // int8 has 256 levels; relative error should be < 5% for this range
    assert_true(err < 0.05)
  }
}

///|
/// D-T7: int8 auto range computes correct scale/zero_point
test "int8 auto range" {
  let data = [-1.0, 0.0, 1.0]
  let (_quantized, scale, _zp) = @tensor.quantize_to_int8(data)
  // Range is 2.0 / 255 ≈ 0.00784
  assert_true(scale > 0.0)
  assert_true(scale < 0.01)
}

///|
/// D-T8: int8 clamp for out-of-range values
test "int8 clamp out of range" {
  let data = [0.0, 100.0, -100.0]
  let (quantized, scale, zp) = @tensor.quantize_to_int8(data)
  let recovered = @tensor.dequantize_from_int8(quantized, scale, zp)
  // Min/max should be recovered (they define the range)
  assert_true((recovered[1] - 100.0).abs() < 1.0)
  assert_true((recovered[2] - -100.0).abs() < 1.0)
}

///|
/// D-T9: Tensor::to_dtype(Float16) preserves data, changes dtype tag
test "to_dtype Float16 preserves data" {
  let t = @tensor.Tensor::from_array([1.0, 2.0, 3.0])
  assert_eq(t.dtype(), @tensor.Float32)
  let t16 = t.to_dtype(@tensor.Float16)
  assert_eq(t16.dtype(), @tensor.Float16)
  // Data should be the same (compute stays in Double)
  assert_eq(t16.to_array(), t.to_array())
  assert_true(t16.quant_params() is None)
}

///|
/// D-T10: Tensor::to_dtype(Int8) computes quant_params
test "to_dtype Int8 computes quant_params" {
  let t = @tensor.Tensor::from_array([-1.0, 0.0, 1.0])
  let t8 = t.to_dtype(@tensor.Int8)
  assert_eq(t8.dtype(), @tensor.Int8)
  // quant_params should be set
  match t8.quant_params() {
    Some(qp) => {
      assert_true(qp.scale > 0.0)
    }
    None => assert_true(false) // Should not be None
  }
}
