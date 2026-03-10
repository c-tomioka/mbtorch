///|
/// Read a little-endian u64 from 8 bytes starting at offset.
fn read_u64_le(data : Bytes, offset : Int) -> UInt64 {
  let mut result = 0UL
  for i = 7; i >= 0; i = i - 1 {
    result = (result << 8) | data[offset + i].to_int().to_uint64()
  }
  result
}

///|
/// Read a little-endian u32 from 4 bytes starting at offset.
fn read_u32_le(data : Bytes, offset : Int) -> UInt {
  let b0 = data[offset].to_int()
  let b1 = data[offset + 1].to_int()
  let b2 = data[offset + 2].to_int()
  let b3 = data[offset + 3].to_int()
  (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).reinterpret_as_uint()
}

///|
/// Convert 4 bytes (IEEE 754 float32 LE) at offset to Double.
fn float32_le_to_double(data : Bytes, offset : Int) -> Double {
  let bits = read_u32_le(data, offset)
  let sign = (bits >> 31) & 1U
  let exponent = (bits >> 23) & 0xFFU
  let mantissa = bits & 0x7FFFFFU
  // Zero
  if exponent == 0U && mantissa == 0U {
    return if sign == 1U { -0.0 } else { 0.0 }
  }
  // Denormalized
  if exponent == 0U {
    let frac = mantissa.reinterpret_as_int().to_double() / 8388608.0 // 2^23
    let value = frac * 1.1754944e-38 // 2^(-126)
    return if sign == 1U { -value } else { value }
  }
  // Infinity / NaN
  if exponent == 0xFFU {
    if mantissa == 0U {
      return if sign == 1U { -1.0 / 0.0 } else { 1.0 / 0.0 }
    }
    return 0.0 / 0.0
  }
  // Normalized
  let frac = 1.0 + mantissa.reinterpret_as_int().to_double() / 8388608.0
  let exp_val = exponent.reinterpret_as_int() - 127
  let mut multiplier = 1.0
  if exp_val >= 0 {
    for _i = 0; _i < exp_val; _i = _i + 1 {
      multiplier = multiplier * 2.0
    }
  } else {
    for _i = 0; _i < -exp_val; _i = _i + 1 {
      multiplier = multiplier / 2.0
    }
  }
  let value = frac * multiplier
  if sign == 1U {
    -value
  } else {
    value
  }
}

///|
/// Read tensor data from raw bytes according to metadata.
fn read_tensor_data(
  data : Bytes,
  meta : SafetensorsMeta,
  header_size : Int,
) -> @tensor.Tensor raise SafetensorsError {
  let byte_count = meta.data_end - meta.data_start
  let element_size = match meta.dtype {
    "F32" => 4
    "F16" => 2
    _ => raise UnsupportedDtype(meta.dtype)
  }
  let num_elements = byte_count / element_size
  // Verify shape matches element count
  let mut expected = 1
  for i = 0; i < meta.shape.length(); i = i + 1 {
    expected = expected * meta.shape[i]
  }
  guard num_elements == expected else {
    let msg = "tensor " +
      meta.name +
      ": expected " +
      expected.to_string() +
      " elements, got " +
      num_elements.to_string()
    raise DataSizeMismatch(msg)
  }
  let values = Array::make(num_elements, 0.0)
  let base_offset = 8 + header_size + meta.data_start
  match meta.dtype {
    "F32" =>
      for i = 0; i < num_elements; i = i + 1 {
        values[i] = float32_le_to_double(data, base_offset + i * 4)
      }
    "F16" =>
      for i = 0; i < num_elements; i = i + 1 {
        values[i] = float16_le_to_double(data, base_offset + i * 2)
      }
    _ => raise UnsupportedDtype(meta.dtype)
  }
  @tensor.Tensor::from_array_with_shape(values, meta.shape.copy())
}

///|
/// Parse the safetensors header and return metadata for each tensor.
pub fn parse_safetensors_header(
  data : Bytes,
) -> Array[SafetensorsMeta] raise SafetensorsError {
  guard data.length() >= 8 else {
    raise InvalidHeader("data too short: need at least 8 bytes")
  }
  let header_size = read_u64_le(data, 0).to_int()
  guard data.length() >= 8 + header_size else {
    let msg = "data too short: header_size=" +
      header_size.to_string() +
      " but data length=" +
      data.length().to_string()
    raise InvalidHeader(msg)
  }
  // Extract header JSON string (ASCII)
  let buf = StringBuilder::new()
  for i = 0; i < header_size; i = i + 1 {
    buf.write_char(data[8 + i].to_int().unsafe_to_char())
  }
  let header_str = buf.to_string()
  // Parse JSON
  let json = @json.parse(header_str) catch {
    _ => raise InvalidHeader("invalid JSON in header")
  }
  // Extract tensor metadata from JSON object
  let obj = match json {
    Json::Object(map) => map
    _ => raise InvalidHeader("header is not a JSON object")
  }
  let metas : Array[SafetensorsMeta] = []
  obj.each(fn(key, value) {
    // Skip __metadata__ key
    if key == "__metadata__" {
      return
    }
    match value {
      Json::Object(tensor_map) => {
        // Extract dtype
        let dtype = match tensor_map.get("dtype") {
          Some(Json::String(s)) => s
          _ => ""
        }
        // Extract shape
        let shape : Array[Int] = []
        match tensor_map.get("shape") {
          Some(Json::Array(arr)) =>
            for i = 0; i < arr.length(); i = i + 1 {
              match arr[i] {
                Json::Number(n, ..) => shape.push(n.to_int())
                _ => ()
              }
            }
          _ => ()
        }
        // Extract data_offsets
        let mut data_start = 0
        let mut data_end = 0
        match tensor_map.get("data_offsets") {
          Some(Json::Array(arr)) =>
            if arr.length() >= 2 {
              match arr[0] {
                Json::Number(n, ..) => data_start = n.to_int()
                _ => ()
              }
              match arr[1] {
                Json::Number(n, ..) => data_end = n.to_int()
                _ => ()
              }
            }
          _ => ()
        }
        metas.push(SafetensorsMeta::{
          name: key,
          dtype,
          shape,
          data_start,
          data_end,
        })
      }
      _ => ()
    }
  })
  metas
}

///|
/// Load all tensors from a safetensors binary blob.
/// Returns an array of (name, Tensor) pairs.
pub fn load_safetensors(
  data : Bytes,
) -> Array[(String, @tensor.Tensor)] raise SafetensorsError {
  let header_size = read_u64_le(data, 0).to_int()
  let metas = parse_safetensors_header(data)
  let result : Array[(String, @tensor.Tensor)] = []
  for i = 0; i < metas.length(); i = i + 1 {
    let meta = metas[i]
    let tensor = read_tensor_data(data, meta, header_size)
    result.push((meta.name, tensor))
  }
  result
}
