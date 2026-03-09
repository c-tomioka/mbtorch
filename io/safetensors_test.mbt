///|
/// Helper: encode a Double as IEEE 754 float32 little-endian (4 bytes).
fn float32_to_le_bytes(value : Double) -> Array[Byte] {
  // Handle zero
  if value == 0.0 {
    return [b'\x00', b'\x00', b'\x00', b'\x00']
  }
  let negative = value < 0.0
  let abs_val = if negative { -value } else { value }
  // Compute exponent and mantissa
  let mut exp = 127
  let mut mantissa_f = abs_val
  if mantissa_f >= 2.0 {
    while mantissa_f >= 2.0 {
      mantissa_f = mantissa_f / 2.0
      exp = exp + 1
    }
  } else if mantissa_f < 1.0 {
    while mantissa_f < 1.0 && exp > 0 {
      mantissa_f = mantissa_f * 2.0
      exp = exp - 1
    }
  }
  // mantissa_f is now in [1.0, 2.0), subtract implicit 1
  let frac = mantissa_f - 1.0
  let mantissa_bits = (frac * 8388608.0 + 0.5).to_int() // 2^23 = 8388608
  let sign_bit = if negative { 1 } else { 0 }
  let bits = (sign_bit << 31) | (exp << 23) | mantissa_bits
  let b0 = (bits & 0xFF).to_byte()
  let b1 = ((bits >> 8) & 0xFF).to_byte()
  let b2 = ((bits >> 16) & 0xFF).to_byte()
  let b3 = ((bits >> 24) & 0xFF).to_byte()
  [b0, b1, b2, b3]
}

///|
/// Helper: convert an ASCII string to Array[Byte].
fn ascii_to_bytes(s : String) -> Array[Byte] {
  let result : Array[Byte] = []
  for i = 0; i < s.length(); i = i + 1 {
    result.push(s[i].to_int().to_byte())
  }
  result
}

///|
/// Helper: build a safetensors binary from JSON header string and tensor data.
fn build_safetensors_bytes(
  header_json : String,
  tensor_data : Array[Byte],
) -> Bytes {
  let header_bytes = ascii_to_bytes(header_json)
  let header_len = header_bytes.length()
  let buf : Array[Byte] = []
  // 8-byte LE u64 for header length
  buf.push((header_len & 0xFF).to_byte())
  buf.push(((header_len >> 8) & 0xFF).to_byte())
  buf.push(((header_len >> 16) & 0xFF).to_byte())
  buf.push(((header_len >> 24) & 0xFF).to_byte())
  buf.push(b'\x00')
  buf.push(b'\x00')
  buf.push(b'\x00')
  buf.push(b'\x00')
  // Copy header
  for i = 0; i < header_len; i = i + 1 {
    buf.push(header_bytes[i])
  }
  // Copy tensor data
  for i = 0; i < tensor_data.length(); i = i + 1 {
    buf.push(tensor_data[i])
  }
  Bytes::from_array(buf)
}

///|
/// Helper: build a complete safetensors binary with named tensors.
/// Each entry is (name, shape, values).
fn build_safetensors_with_tensors(
  tensors : Array[(String, Array[Int], Array[Double])],
) -> Bytes {
  // First pass: compute data layout and build JSON header
  let mut offset = 0
  let header_parts : Array[String] = []
  let all_bytes : Array[Byte] = []
  for i = 0; i < tensors.length(); i = i + 1 {
    let (name, shape, values) = tensors[i]
    let data_size = values.length() * 4 // float32 = 4 bytes
    let data_start = offset
    let data_end = offset + data_size
    // Build shape string
    let shape_strs : Array[String] = []
    for j = 0; j < shape.length(); j = j + 1 {
      shape_strs.push(shape[j].to_string())
    }
    let shape_str = shape_strs.join(",")
    let start_str = data_start.to_string()
    let end_str = data_end.to_string()
    // Build JSON entry without quotes in interpolation
    let part = "\"" +
      name +
      "\":{\"dtype\":\"F32\",\"shape\":[" +
      shape_str +
      "],\"data_offsets\":[" +
      start_str +
      "," +
      end_str +
      "]}"
    header_parts.push(part)
    // Build float32 LE data
    for j = 0; j < values.length(); j = j + 1 {
      let fb = float32_to_le_bytes(values[j])
      all_bytes.push(fb[0])
      all_bytes.push(fb[1])
      all_bytes.push(fb[2])
      all_bytes.push(fb[3])
    }
    offset = data_end
  }
  let header_json = "{" + header_parts.join(",") + "}"
  build_safetensors_bytes(header_json, all_bytes)
}

///|
/// S1: Parse header of a single F32 tensor
test "parse_safetensors_header single tensor" {
  let data = build_safetensors_with_tensors([
    ("weight", [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
  ])
  let metas = @io.parse_safetensors_header(data)
  assert_eq(metas.length(), 1)
  assert_eq(metas[0].name, "weight")
  assert_eq(metas[0].dtype, "F32")
  assert_eq(metas[0].shape, [2, 3])
  assert_eq(metas[0].data_end - metas[0].data_start, 24) // 6 * 4 bytes
}

///|
/// S4: Load a 1D float32 tensor and verify values
test "load_safetensors 1d float32 tensor" {
  let values = [1.0, 2.0, 3.0]
  let data = build_safetensors_with_tensors([("bias", [3], values)])
  let result = @io.load_safetensors(data)
  assert_eq(result.length(), 1)
  let (name, tensor) = result[0]
  assert_eq(name, "bias")
  assert_eq(tensor.shape(), [3])
  // float32 precision: compare with tolerance
  let eps = 1.0e-6
  let arr = tensor.to_array()
  assert_true((arr[0] - 1.0).abs() < eps)
  assert_true((arr[1] - 2.0).abs() < eps)
  assert_true((arr[2] - 3.0).abs() < eps)
}

///|
/// S5: Load a 2D float32 tensor and verify shape and values
test "load_safetensors 2d float32 tensor" {
  let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let data = build_safetensors_with_tensors([("weight", [2, 3], values)])
  let result = @io.load_safetensors(data)
  assert_eq(result.length(), 1)
  let (name, tensor) = result[0]
  assert_eq(name, "weight")
  assert_eq(tensor.shape(), [2, 3])
  let arr = tensor.to_array()
  let eps = 1.0e-6
  assert_true((arr[0] - 1.0).abs() < eps)
  assert_true((arr[3] - 4.0).abs() < eps)
  assert_true((arr[5] - 6.0).abs() < eps)
}

///|
/// S8: Reject unsupported dtype (F64)
test "load_safetensors rejects unsupported dtype" {
  // Build a safetensors with F64 dtype manually
  let header_json =
    #|{"x":{"dtype":"F64","shape":[2],"data_offsets":[0,16]}}
  let fake_data : Array[Byte] = Array::make(16, b'\x00')
  let data = build_safetensors_bytes(header_json, fake_data)
  let mut got_error = false
  try {
    let _ = @io.load_safetensors(data)
  } catch {
    @io.UnsupportedDtype(_) => got_error = true
    _ => ()
  }
  assert_true(got_error)
}
