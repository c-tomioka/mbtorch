///|
/// DTO for tensor metadata in binary .mbt format
priv struct BinaryTensorMeta {
  shape : Array[Int]
  dtype : String
  offset : Int
  length : Int
  scale : Double?
  zero_point : Int?
} derive(ToJson, FromJson)

///|
/// DTO for layer metadata in binary .mbt format
priv struct BinaryLayerMeta {
  name : String
  layer_type : String
  in_features : Int
  out_features : Int
  weight : BinaryTensorMeta
  bias : BinaryTensorMeta
} derive(ToJson, FromJson)

///|
/// Top-level DTO for binary .mbt header
priv struct BinaryMbtHeader {
  format : String
  version : Int
  layers : Array[BinaryLayerMeta]
} derive(ToJson, FromJson)

///|
/// Convert a Double to IEEE 754 float32 little-endian (4 bytes).
fn double_to_float32_le(value : Double) -> Bytes {
  // Handle zero
  if value == 0.0 {
    // Check for -0.0
    if 1.0 / value < 0.0 {
      return Bytes::from_array([b'\x00', b'\x00', b'\x00', b'\x80'])
    }
    return Bytes::from_array([b'\x00', b'\x00', b'\x00', b'\x00'])
  }
  let negative = value < 0.0
  let abs_val = if negative { -value } else { value }
  // Handle infinity
  if abs_val > 3.4028235e38 {
    let sign_byte : Byte = if negative { b'\x80' } else { b'\x00' }
    // 0x7F800000 for +inf, 0xFF800000 for -inf
    return Bytes::from_array([b'\x00', b'\x00', b'\x80', sign_byte | b'\x7F'])
  }
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
  // Denormalized number
  if exp == 0 {
    let frac = abs_val / 1.1754944e-38 // 2^(-126)
    let mantissa_bits = (frac * 8388608.0 + 0.5).to_int() // 2^23
    let sign_bit = if negative { 1 } else { 0 }
    let bits = (sign_bit << 31) | mantissa_bits
    let b0 = (bits & 0xFF).to_byte()
    let b1 = ((bits >> 8) & 0xFF).to_byte()
    let b2 = ((bits >> 16) & 0xFF).to_byte()
    let b3 = ((bits >> 24) & 0xFF).to_byte()
    return Bytes::from_array([b0, b1, b2, b3])
  }
  // Normalized number
  let frac = mantissa_f - 1.0
  let mantissa_bits = (frac * 8388608.0 + 0.5).to_int() // 2^23
  let sign_bit = if negative { 1 } else { 0 }
  let bits = (sign_bit << 31) | (exp << 23) | mantissa_bits
  let b0 = (bits & 0xFF).to_byte()
  let b1 = ((bits >> 8) & 0xFF).to_byte()
  let b2 = ((bits >> 16) & 0xFF).to_byte()
  let b3 = ((bits >> 24) & 0xFF).to_byte()
  Bytes::from_array([b0, b1, b2, b3])
}

///|
/// Convert an ASCII string to Array[Byte].
fn string_to_bytes(s : String) -> Array[Byte] {
  let result : Array[Byte] = []
  for i = 0; i < s.length(); i = i + 1 {
    result.push(s[i].to_int().to_byte())
  }
  result
}

///|
/// Serialize named Linear layers to binary .mbt format.
pub fn serialize_model_to_binary(
  layers : Array[(String, @nn.Linear)]
) -> Bytes {
  // First pass: build tensor buffer and metadata
  let tensor_buf : Array[Byte] = []
  let layer_metas : Array[BinaryLayerMeta] = []
  for i = 0; i < layers.length(); i = i + 1 {
    let (name, layer) = layers[i]
    let w = layer.weight.data()
    let b = layer.bias.data()
    let ws = w.shape()
    let bs = b.shape()
    let w_data = w.to_array()
    let b_data = b.to_array()
    // Weight tensor
    let w_offset = tensor_buf.length()
    for j = 0; j < w_data.length(); j = j + 1 {
      let bytes = double_to_float32_le(w_data[j])
      tensor_buf.push(bytes[0])
      tensor_buf.push(bytes[1])
      tensor_buf.push(bytes[2])
      tensor_buf.push(bytes[3])
    }
    let w_length = tensor_buf.length() - w_offset
    // Bias tensor
    let b_offset = tensor_buf.length()
    for j = 0; j < b_data.length(); j = j + 1 {
      let bytes = double_to_float32_le(b_data[j])
      tensor_buf.push(bytes[0])
      tensor_buf.push(bytes[1])
      tensor_buf.push(bytes[2])
      tensor_buf.push(bytes[3])
    }
    let b_length = tensor_buf.length() - b_offset
    layer_metas.push({
      name,
      layer_type: "Linear",
      in_features: ws[0],
      out_features: ws[1],
      weight: { shape: ws, dtype: "F32", offset: w_offset, length: w_length, scale: None, zero_point: None },
      bias: { shape: bs, dtype: "F32", offset: b_offset, length: b_length, scale: None, zero_point: None },
    })
  }
  // Build metadata JSON
  let header : BinaryMbtHeader = {
    format: "mbtorch-binary",
    version: 1,
    layers: layer_metas,
  }
  let meta_str = header.to_json().stringify()
  let meta_bytes = string_to_bytes(meta_str)
  let meta_len = meta_bytes.length()
  // Build final binary
  let result : Array[Byte] = []
  // Magic: "MBTM"
  result.push(b'M')
  result.push(b'B')
  result.push(b'T')
  result.push(b'M')
  // Version: 1 (u32 LE)
  result.push(b'\x01')
  result.push(b'\x00')
  result.push(b'\x00')
  result.push(b'\x00')
  // Metadata size (u32 LE)
  result.push((meta_len & 0xFF).to_byte())
  result.push(((meta_len >> 8) & 0xFF).to_byte())
  result.push(((meta_len >> 16) & 0xFF).to_byte())
  result.push(((meta_len >> 24) & 0xFF).to_byte())
  // Metadata JSON
  for i = 0; i < meta_len; i = i + 1 {
    result.push(meta_bytes[i])
  }
  // Tensor buffer
  for i = 0; i < tensor_buf.length(); i = i + 1 {
    result.push(tensor_buf[i])
  }
  Bytes::from_array(result)
}

///|
/// Deserialize binary .mbt format back to named Linear layers.
pub fn deserialize_model_from_binary(
  data : Bytes
) -> Array[(String, @nn.Linear)] raise MbtFormatError {
  // Minimum size: 12 bytes header
  guard data.length() >= 12 else {
    raise InvalidFormat("binary .mbt data too short: need at least 12 bytes")
  }
  // Validate magic "MBTM"
  guard data[0] == b'M' && data[1] == b'B' && data[2] == b'T' && data[3] == b'M' else {
    raise InvalidFormat("invalid magic bytes: expected \"MBTM\"")
  }
  // Read version (u32 LE)
  let version = read_u32_le(data, 4).reinterpret_as_int()
  guard version == 1 else { raise UnsupportedVersion(version) }
  // Read metadata size (u32 LE)
  let meta_size = read_u32_le(data, 8).reinterpret_as_int()
  guard data.length() >= 12 + meta_size else {
    raise InvalidFormat(
      "data too short for metadata: need \{12 + meta_size} bytes, got \{data.length()}",
    )
  }
  // Extract metadata JSON string
  let buf = StringBuilder::new()
  for i = 0; i < meta_size; i = i + 1 {
    buf.write_char(data[12 + i].to_int().unsafe_to_char())
  }
  let meta_str = buf.to_string()
  let json = @json.parse(meta_str) catch {
    _ => raise InvalidFormat("failed to parse metadata JSON")
  }
  let header : BinaryMbtHeader = @json.from_json(json) catch {
    _ => raise InvalidFormat("metadata JSON does not match binary .mbt format")
  }
  // Validate format
  guard header.format == "mbtorch-binary" else {
    raise InvalidFormat(
      "expected format \"mbtorch-binary\", got \"\{header.format}\"",
    )
  }
  // Tensor buffer starts at offset 12 + meta_size
  let buf_start = 12 + meta_size
  // Reconstruct layers
  let result : Array[(String, @nn.Linear)] = []
  for i = 0; i < header.layers.length(); i = i + 1 {
    let ld = header.layers[i]
    guard ld.layer_type == "Linear" else {
      raise InvalidFormat("unsupported layer type: \{ld.layer_type}")
    }
    // Read weight tensor
    let w_meta = ld.weight
    let w_values = decode_tensor_from_buf(
      data, buf_start + w_meta.offset, w_meta.length,
      w_meta.dtype, w_meta.scale, w_meta.zero_point,
    )
    let weight = @tensor.Tensor::from_array_with_shape(w_values, w_meta.shape)
    // Read bias tensor
    let b_meta = ld.bias
    let b_values = decode_tensor_from_buf(
      data, buf_start + b_meta.offset, b_meta.length,
      b_meta.dtype, b_meta.scale, b_meta.zero_point,
    )
    let bias = @tensor.Tensor::from_array_with_shape(b_values, b_meta.shape)
    let linear = @nn.Linear::from_tensors(weight, bias)
    result.push((ld.name, linear))
  }
  result
}

///|
/// Serialize named Linear layers to binary .mbt format with specified dtype.
/// dtype: "F32" (4 bytes/elem), "F16" (2 bytes/elem), "I8" (1 byte/elem)
pub fn serialize_model_to_binary_with_dtype(
  layers : Array[(String, @nn.Linear)],
  dtype : String,
) -> Bytes {
  let tensor_buf : Array[Byte] = []
  let layer_metas : Array[BinaryLayerMeta] = []
  for i = 0; i < layers.length(); i = i + 1 {
    let (name, layer) = layers[i]
    let w = layer.weight.data()
    let b = layer.bias.data()
    let ws = w.shape()
    let bs = b.shape()
    let w_data = w.to_array()
    let b_data = b.to_array()
    // Weight tensor
    let w_offset = tensor_buf.length()
    let (w_scale, w_zp) = encode_tensor_to_buf(tensor_buf, w_data, dtype)
    let w_length = tensor_buf.length() - w_offset
    // Bias tensor (always F32 for int8 to preserve precision)
    let b_offset = tensor_buf.length()
    let bias_dtype = if dtype == "I8" { "F32" } else { dtype }
    let (b_scale, b_zp) = encode_tensor_to_buf(tensor_buf, b_data, bias_dtype)
    let b_length = tensor_buf.length() - b_offset
    layer_metas.push({
      name,
      layer_type: "Linear",
      in_features: ws[0],
      out_features: ws[1],
      weight: { shape: ws, dtype, offset: w_offset, length: w_length, scale: w_scale, zero_point: w_zp },
      bias: { shape: bs, dtype: bias_dtype, offset: b_offset, length: b_length, scale: b_scale, zero_point: b_zp },
    })
  }
  let header : BinaryMbtHeader = {
    format: "mbtorch-binary",
    version: 1,
    layers: layer_metas,
  }
  assemble_binary(header, tensor_buf)
}

///|
/// Encode tensor data into buffer based on dtype. Returns (scale?, zero_point?) for I8.
fn encode_tensor_to_buf(
  buf : Array[Byte],
  data : Array[Double],
  dtype : String,
) -> (Double?, Int?) {
  match dtype {
    "F32" => {
      for j = 0; j < data.length(); j = j + 1 {
        let bytes = double_to_float32_le(data[j])
        buf.push(bytes[0])
        buf.push(bytes[1])
        buf.push(bytes[2])
        buf.push(bytes[3])
      }
      (None, None)
    }
    "F16" => {
      for j = 0; j < data.length(); j = j + 1 {
        let bytes = double_to_float16_le(data[j])
        buf.push(bytes[0])
        buf.push(bytes[1])
      }
      (None, None)
    }
    "I8" => {
      let (quantized, scale, zp) = @tensor.quantize_to_int8(data)
      for j = 0; j < quantized.length(); j = j + 1 {
        buf.push(quantized[j].to_byte())
      }
      (Some(scale), Some(zp))
    }
    _ => {
      // Fallback to F32
      for j = 0; j < data.length(); j = j + 1 {
        let bytes = double_to_float32_le(data[j])
        buf.push(bytes[0])
        buf.push(bytes[1])
        buf.push(bytes[2])
        buf.push(bytes[3])
      }
      (None, None)
    }
  }
}

///|
/// Decode tensor values from binary buffer based on dtype.
fn decode_tensor_from_buf(
  data : Bytes,
  offset : Int,
  num_bytes : Int,
  dtype : String,
  scale : Double?,
  zero_point : Int?,
) -> Array[Double] {
  match dtype {
    "F32" => {
      let n = num_bytes / 4
      let values = Array::make(n, 0.0)
      for j = 0; j < n; j = j + 1 {
        values[j] = float32_le_to_double(data, offset + j * 4)
      }
      values
    }
    "F16" => {
      let n = num_bytes / 2
      let values = Array::make(n, 0.0)
      for j = 0; j < n; j = j + 1 {
        values[j] = float16_le_to_double(data, offset + j * 2)
      }
      values
    }
    "I8" => {
      let s = match scale {
        Some(v) => v
        None => 1.0
      }
      let zp = match zero_point {
        Some(v) => v
        None => 0
      }
      let values = Array::make(num_bytes, 0.0)
      for j = 0; j < num_bytes; j = j + 1 {
        values[j] = uint8_to_double(data[offset + j], s, zp)
      }
      values
    }
    _ => {
      // Fallback to F32
      let n = num_bytes / 4
      let values = Array::make(n, 0.0)
      for j = 0; j < n; j = j + 1 {
        values[j] = float32_le_to_double(data, offset + j * 4)
      }
      values
    }
  }
}

///|
/// Assemble binary .mbt file from header and tensor buffer.
fn assemble_binary(header : BinaryMbtHeader, tensor_buf : Array[Byte]) -> Bytes {
  let meta_str = header.to_json().stringify()
  let meta_bytes = string_to_bytes(meta_str)
  let meta_len = meta_bytes.length()
  let result : Array[Byte] = []
  result.push(b'M')
  result.push(b'B')
  result.push(b'T')
  result.push(b'M')
  result.push(b'\x01')
  result.push(b'\x00')
  result.push(b'\x00')
  result.push(b'\x00')
  result.push((meta_len & 0xFF).to_byte())
  result.push(((meta_len >> 8) & 0xFF).to_byte())
  result.push(((meta_len >> 16) & 0xFF).to_byte())
  result.push(((meta_len >> 24) & 0xFF).to_byte())
  for i = 0; i < meta_len; i = i + 1 {
    result.push(meta_bytes[i])
  }
  for i = 0; i < tensor_buf.length(); i = i + 1 {
    result.push(tensor_buf[i])
  }
  Bytes::from_array(result)
}

///|
test "double_to_float32_le roundtrip" {
  let values = [0.0, 1.0, -1.0, 3.14159, 1.0e-10, 1.0e10, -0.5, 42.0]
  for i = 0; i < values.length(); i = i + 1 {
    let bytes = double_to_float32_le(values[i])
    let recovered = float32_le_to_double(bytes, 0)
    let diff = (values[i] - recovered).abs()
    let tol = if values[i] == 0.0 {
      1.0e-38
    } else {
      values[i].abs() * 1.0e-6
    }
    assert_true(diff < tol)
  }
}
