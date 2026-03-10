// Protobuf encoding primitives for ONNX export and test fixtures

///|
/// Encode a UInt64 as a protobuf varint.
fn encode_varint(value : UInt64) -> Array[Byte] {
  let result : Array[Byte] = []
  let mut v = value
  while v >= 0x80UL {
    result.push(((v.to_int() & 0x7F) | 0x80).to_byte())
    v = v >> 7
  }
  result.push((v.to_int() & 0x7F).to_byte())
  result
}

///|
/// Encode a protobuf field tag.
fn encode_tag(field_num : Int, wire_type : Int) -> Array[Byte] {
  encode_varint(((field_num << 3) | wire_type).to_uint64())
}

///|
/// Append all bytes from source to target.
fn append_all(target : Array[Byte], source : Array[Byte]) -> Unit {
  for i = 0; i < source.length(); i = i + 1 {
    target.push(source[i])
  }
}

///|
/// Encode a length-delimited field (wire type 2).
fn encode_len_delimited(field_num : Int, content : Array[Byte]) -> Array[Byte] {
  let result : Array[Byte] = []
  append_all(result, encode_tag(field_num, 2))
  append_all(result, encode_varint(content.length().to_uint64()))
  append_all(result, content)
  result
}

///|
/// Encode a varint field (wire type 0).
fn encode_varint_field(field_num : Int, value : UInt64) -> Array[Byte] {
  let result : Array[Byte] = []
  append_all(result, encode_tag(field_num, 0))
  append_all(result, encode_varint(value))
  result
}

///|
/// Encode a string field (wire type 2).
fn encode_string_field(field_num : Int, s : String) -> Array[Byte] {
  let bytes : Array[Byte] = []
  for i = 0; i < s.length(); i = i + 1 {
    bytes.push(s[i].to_int().to_byte())
  }
  encode_len_delimited(field_num, bytes)
}

///|
/// Encode a fixed32 field (wire type 5).
fn encode_fixed32_field(field_num : Int, b : Array[Byte]) -> Array[Byte] {
  let result : Array[Byte] = []
  append_all(result, encode_tag(field_num, 5))
  append_all(result, b)
  result
}

///|
/// Encode a Double as IEEE 754 float32 LE (4 bytes).
fn onnx_float32_to_le_bytes(value : Double) -> Array[Byte] {
  if value == 0.0 {
    return [b'\x00', b'\x00', b'\x00', b'\x00']
  }
  let negative = value < 0.0
  let abs_val = if negative { -value } else { value }
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
  let frac = mantissa_f - 1.0
  let mantissa_bits = (frac * 8388608.0 + 0.5).to_int()
  let sign_bit = if negative { 1 } else { 0 }
  let bits = (sign_bit << 31) | (exp << 23) | mantissa_bits
  [
    (bits & 0xFF).to_byte(),
    ((bits >> 8) & 0xFF).to_byte(),
    ((bits >> 16) & 0xFF).to_byte(),
    ((bits >> 24) & 0xFF).to_byte(),
  ]
}

///|
/// Build a TensorProto with float_data (packed float32).
/// TensorProto fields: 1=dims(packed int64), 2=data_type(varint), 4=float_data(packed f32), 8=name
fn build_tensor_proto(
  name : String,
  shape : Array[Int],
  values : Array[Double],
) -> Array[Byte] {
  let content : Array[Byte] = []
  // Field 1: dims (packed int64 varints)
  let dims_content : Array[Byte] = []
  for i = 0; i < shape.length(); i = i + 1 {
    append_all(dims_content, encode_varint(shape[i].to_uint64()))
  }
  append_all(content, encode_len_delimited(1, dims_content))
  // Field 2: data_type = 1 (FLOAT)
  append_all(content, encode_varint_field(2, 1UL))
  // Field 4: float_data (packed float32 LE bytes)
  let float_content : Array[Byte] = []
  for i = 0; i < values.length(); i = i + 1 {
    append_all(float_content, onnx_float32_to_le_bytes(values[i]))
  }
  append_all(content, encode_len_delimited(4, float_content))
  // Field 8: name
  append_all(content, encode_string_field(8, name))
  content
}

///|
/// Build a TensorProto with dtype=10 (FLOAT16) using raw_data (2 bytes per value).
fn build_tensor_proto_f16(
  name : String,
  shape : Array[Int],
  values : Array[Double],
) -> Array[Byte] {
  let content : Array[Byte] = []
  // Field 1: dims (packed int64 varints)
  let dims_content : Array[Byte] = []
  for i = 0; i < shape.length(); i = i + 1 {
    append_all(dims_content, encode_varint(shape[i].to_uint64()))
  }
  append_all(content, encode_len_delimited(1, dims_content))
  // Field 2: data_type = 10 (FLOAT16)
  append_all(content, encode_varint_field(2, 10UL))
  // Field 9: raw_data (packed float16 LE, 2 bytes per value)
  let raw_content : Array[Byte] = []
  for i = 0; i < values.length(); i = i + 1 {
    let bytes = double_to_float16_le(values[i])
    raw_content.push(bytes[0])
    raw_content.push(bytes[1])
  }
  append_all(content, encode_len_delimited(9, raw_content))
  // Field 8: name
  append_all(content, encode_string_field(8, name))
  content
}

///|
/// Build a NodeProto with integer-only attributes.
/// NodeProto fields: 1=input(repeated string), 2=output(repeated string), 4=op_type, 5=attribute
fn build_node_proto(
  op_type : String,
  inputs : Array[String],
  outputs : Array[String],
  int_attrs : Array[(String, Int64)],
) -> Array[Byte] {
  let content : Array[Byte] = []
  // Field 1: inputs (repeated string, each as separate LEN field)
  for i = 0; i < inputs.length(); i = i + 1 {
    append_all(content, encode_string_field(1, inputs[i]))
  }
  // Field 2: outputs
  for i = 0; i < outputs.length(); i = i + 1 {
    append_all(content, encode_string_field(2, outputs[i]))
  }
  // Field 4: op_type
  append_all(content, encode_string_field(4, op_type))
  // Field 5: attributes (each as sub-message)
  for i = 0; i < int_attrs.length(); i = i + 1 {
    let (attr_name, attr_val) = int_attrs[i]
    let attr_content : Array[Byte] = []
    // AttributeProto field 1: name
    append_all(attr_content, encode_string_field(1, attr_name))
    // AttributeProto field 3: i (int64 varint)
    append_all(
      attr_content,
      encode_varint_field(3, attr_val.to_int().to_uint64()),
    )
    // AttributeProto field 20: type = 2 (INT)
    append_all(attr_content, encode_varint_field(20, 2UL))
    append_all(content, encode_len_delimited(5, attr_content))
  }
  content
}

///|
/// Build a NodeProto with full OnnxAttribute support (int, float, ints).
/// AttributeProto fields: 1=name, 2=f(float32), 3=i(int64), 7=ints(packed), 20=type
fn build_node_proto_full(
  op_type : String,
  inputs : Array[String],
  outputs : Array[String],
  attrs : Array[OnnxAttribute],
) -> Array[Byte] {
  let content : Array[Byte] = []
  for i = 0; i < inputs.length(); i = i + 1 {
    append_all(content, encode_string_field(1, inputs[i]))
  }
  for i = 0; i < outputs.length(); i = i + 1 {
    append_all(content, encode_string_field(2, outputs[i]))
  }
  append_all(content, encode_string_field(4, op_type))
  for i = 0; i < attrs.length(); i = i + 1 {
    let attr = attrs[i]
    let attr_content : Array[Byte] = []
    // Field 1: name
    append_all(attr_content, encode_string_field(1, attr.name))
    if attr.ints.length() > 0 {
      // Field 8: ints (repeated int64, packed)
      // Note: field 7 is floats, field 8 is ints in AttributeProto
      let ints_content : Array[Byte] = []
      for j = 0; j < attr.ints.length(); j = j + 1 {
        append_all(ints_content, encode_varint(attr.ints[j].to_uint64()))
      }
      append_all(attr_content, encode_len_delimited(8, ints_content))
      // Field 20: type = 7 (INTS)
      append_all(attr_content, encode_varint_field(20, 7UL))
    } else if attr.float_value != 0.0 {
      // Field 2: f (fixed32, wire type 5)
      append_all(
        attr_content,
        encode_fixed32_field(2, onnx_float32_to_le_bytes(attr.float_value)),
      )
      // Field 20: type = 1 (FLOAT)
      append_all(attr_content, encode_varint_field(20, 1UL))
    } else if attr.int_value != 0L {
      // Field 3: i (varint)
      append_all(
        attr_content,
        encode_varint_field(3, attr.int_value.to_int().to_uint64()),
      )
      // Field 20: type = 2 (INT)
      append_all(attr_content, encode_varint_field(20, 2UL))
    } else {
      // int_value == 0 but may still be intentional INT attr
      // Encode as INT with value 0
      append_all(attr_content, encode_varint_field(3, 0UL))
      // Field 20: type = 2 (INT)
      append_all(attr_content, encode_varint_field(20, 2UL))
    }
    append_all(content, encode_len_delimited(5, attr_content))
  }
  content
}

///|
/// Build a ValueInfoProto with name and tensor(float) type.
/// Required by onnxruntime for graph inputs/outputs.
fn build_value_info(name : String) -> Array[Byte] {
  let content : Array[Byte] = []
  // ValueInfoProto field 1: name
  append_all(content, encode_string_field(1, name))
  // ValueInfoProto field 2: type (TypeProto)
  //   TypeProto field 1: tensor_type (Tensor_TypeProto)
  //     Tensor_TypeProto field 1: elem_type = 1 (FLOAT)
  let tensor_type : Array[Byte] = []
  append_all(tensor_type, encode_varint_field(1, 1UL))
  let type_proto : Array[Byte] = []
  append_all(type_proto, encode_len_delimited(1, tensor_type))
  append_all(content, encode_len_delimited(2, type_proto))
  content
}

///|
/// Build a GraphProto.
/// GraphProto fields: 1=node, 5=initializer, 11=input, 12=output
fn build_graph_proto(
  nodes : Array[Array[Byte]],
  inits : Array[Array[Byte]],
  inputs : Array[String],
  outputs : Array[String],
) -> Array[Byte] {
  let content : Array[Byte] = []
  // Field 1: nodes (each as LEN)
  for i = 0; i < nodes.length(); i = i + 1 {
    append_all(content, encode_len_delimited(1, nodes[i]))
  }
  // Field 5: initializers (each as LEN)
  for i = 0; i < inits.length(); i = i + 1 {
    append_all(content, encode_len_delimited(5, inits[i]))
  }
  // Field 11: inputs (each as ValueInfoProto LEN)
  for i = 0; i < inputs.length(); i = i + 1 {
    append_all(content, encode_len_delimited(11, build_value_info(inputs[i])))
  }
  // Field 12: outputs (each as ValueInfoProto LEN)
  for i = 0; i < outputs.length(); i = i + 1 {
    append_all(content, encode_len_delimited(12, build_value_info(outputs[i])))
  }
  content
}

///|
/// Build a ModelProto with opset_import.
/// ModelProto fields: 7=graph (LEN), 8=opset_import (LEN sub-message)
/// OpSetIdProto field 2: version (varint)
fn build_model_proto_with_opset(
  graph : Array[Byte],
  opset_version : Int,
) -> Bytes {
  let content : Array[Byte] = []
  // Field 1: ir_version (varint) — required by onnxruntime
  append_all(content, encode_varint_field(1, 7UL))
  // Field 7: graph
  append_all(content, encode_len_delimited(7, graph))
  // Field 8: opset_import (sub-message with field 2 = version)
  let opset_content : Array[Byte] = []
  append_all(
    opset_content,
    encode_varint_field(2, opset_version.to_uint64()),
  )
  append_all(content, encode_len_delimited(8, opset_content))
  Bytes::from_array(content)
}
