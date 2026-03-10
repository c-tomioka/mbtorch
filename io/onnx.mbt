///|
/// Decode a protobuf varint. Returns (value, new_offset).
fn read_varint(data : Bytes, offset : Int) -> (UInt64, Int) {
  let mut result = 0UL
  let mut shift = 0
  let mut pos = offset
  while pos < data.length() {
    let byte = data[pos].to_int()
    result = result | ((byte & 0x7F).to_uint64() << shift)
    pos = pos + 1
    if (byte & 0x80) == 0 {
      break
    }
    shift = shift + 7
  }
  (result, pos)
}

///|
/// Skip an unknown field based on wire type. Returns new offset.
fn skip_field(data : Bytes, offset : Int, wire_type : Int) -> Int {
  match wire_type {
    0 => {
      // varint: read and discard
      let (_, new_pos) = read_varint(data, offset)
      new_pos
    }
    1 =>
      // 64-bit fixed
      offset + 8
    2 => {
      // length-delimited
      let (len, new_pos) = read_varint(data, offset)
      new_pos + len.to_int()
    }
    5 =>
      // 32-bit fixed
      offset + 4
    _ => offset
  }
}

///|
/// Read a UTF-8/ASCII string from bytes.
fn parse_pb_string(data : Bytes, offset : Int, len : Int) -> String {
  let buf = StringBuilder::new()
  for i = 0; i < len; i = i + 1 {
    buf.write_char(data[offset + i].to_int().unsafe_to_char())
  }
  buf.to_string()
}

///|
/// Decode packed int64 varints from a length-delimited field.
fn parse_packed_int64(data : Bytes, offset : Int, len : Int) -> Array[Int] {
  let result : Array[Int] = []
  let end = offset + len
  let mut pos = offset
  while pos < end {
    let (val, new_pos) = read_varint(data, pos)
    result.push(val.to_int())
    pos = new_pos
  }
  result
}

///|
/// Decode packed float32 values from a length-delimited field.
/// Reuses float32_le_to_double from safetensors.mbt (same package).
fn parse_packed_float32(data : Bytes, offset : Int, len : Int) -> Array[Double] {
  let count = len / 4
  let result = Array::make(count, 0.0)
  for i = 0; i < count; i = i + 1 {
    result[i] = float32_le_to_double(data, offset + i * 4)
  }
  result
}

///|
/// Parse TensorProto from a length-delimited sub-message.
/// Fields: 1=dims, 2=data_type, 4=float_data, 8=name, 9=raw_data
fn parse_tensor_proto(
  data : Bytes,
  offset : Int,
  len : Int,
) -> OnnxTensor raise OnnxError {
  let end = offset + len
  let mut pos = offset
  let mut name = ""
  let mut dtype = 0
  let dims : Array[Int] = []
  let float_data : Array[Double] = []
  let mut raw_data_offset = -1
  let mut raw_data_len = 0
  while pos < end {
    let (tag_val, tag_end) = read_varint(data, pos)
    let field_num = (tag_val >> 3).to_int()
    let wire_type = (tag_val & 0x7UL).to_int()
    pos = tag_end
    match field_num {
      1 => {
        // dims: packed int64
        guard wire_type == 2 else {
          pos = skip_field(data, pos, wire_type)
          continue
        }
        let (field_len, data_start) = read_varint(data, pos)
        let parsed_dims = parse_packed_int64(
          data,
          data_start,
          field_len.to_int(),
        )
        for i = 0; i < parsed_dims.length(); i = i + 1 {
          dims.push(parsed_dims[i])
        }
        pos = data_start + field_len.to_int()
      }
      2 => {
        // data_type: varint
        let (val, new_pos) = read_varint(data, pos)
        dtype = val.to_int()
        pos = new_pos
      }
      4 => {
        // float_data: packed float32
        guard wire_type == 2 else {
          pos = skip_field(data, pos, wire_type)
          continue
        }
        let (field_len, data_start) = read_varint(data, pos)
        let parsed = parse_packed_float32(data, data_start, field_len.to_int())
        for i = 0; i < parsed.length(); i = i + 1 {
          float_data.push(parsed[i])
        }
        pos = data_start + field_len.to_int()
      }
      8 => {
        // name: string
        let (field_len, data_start) = read_varint(data, pos)
        name = parse_pb_string(data, data_start, field_len.to_int())
        pos = data_start + field_len.to_int()
      }
      9 => {
        // raw_data: bytes
        let (field_len, data_start) = read_varint(data, pos)
        raw_data_offset = data_start
        raw_data_len = field_len.to_int()
        pos = data_start + field_len.to_int()
      }
      _ => pos = skip_field(data, pos, wire_type)
    }
  }
  // Validate dtype: 1=FLOAT, 10=FLOAT16
  guard dtype == 1 || dtype == 10 else {
    raise UnsupportedDtype(
      "tensor " + name + ": expected FLOAT (1) or FLOAT16 (10), got " + dtype.to_string(),
    )
  }
  // Get float values from either float_data or raw_data
  let values = if dtype == 10 {
    // FLOAT16: always in raw_data (2 bytes per element)
    if raw_data_offset >= 0 {
      let count = raw_data_len / 2
      let result = Array::make(count, 0.0)
      for i = 0; i < count; i = i + 1 {
        result[i] = float16_le_to_double(data, raw_data_offset + i * 2)
      }
      result
    } else {
      []
    }
  } else if float_data.length() > 0 {
    float_data
  } else if raw_data_offset >= 0 {
    parse_packed_float32(data, raw_data_offset, raw_data_len)
  } else {
    []
  }
  OnnxTensor::{ name, dtype, shape: dims, data: values }
}

///|
/// Parse AttributeProto from a length-delimited sub-message.
/// Fields: 1=name, 2=f(I32/wire5), 3=i(varint)
fn parse_attribute_proto(
  data : Bytes,
  offset : Int,
  len : Int,
) -> OnnxAttribute {
  let end = offset + len
  let mut pos = offset
  let mut name = ""
  let mut int_value = 0L
  let mut float_value = 0.0
  let ints : Array[Int] = []
  while pos < end {
    let (tag_val, tag_end) = read_varint(data, pos)
    let field_num = (tag_val >> 3).to_int()
    let wire_type = (tag_val & 0x7UL).to_int()
    pos = tag_end
    match field_num {
      1 => {
        // name: string
        let (field_len, data_start) = read_varint(data, pos)
        name = parse_pb_string(data, data_start, field_len.to_int())
        pos = data_start + field_len.to_int()
      }
      2 => {
        // f: float (wire type 5 = fixed32)
        guard wire_type == 5 else {
          pos = skip_field(data, pos, wire_type)
          continue
        }
        float_value = float32_le_to_double(data, pos)
        pos = pos + 4
      }
      3 => {
        // i: int64 (varint)
        let (val, new_pos) = read_varint(data, pos)
        int_value = val.to_int().to_int64()
        pos = new_pos
      }
      7 => {
        // ints: repeated int64 (packed or individual)
        if wire_type == 2 {
          // packed
          let (field_len, data_start) = read_varint(data, pos)
          let parsed = parse_packed_int64(data, data_start, field_len.to_int())
          for j = 0; j < parsed.length(); j = j + 1 {
            ints.push(parsed[j])
          }
          pos = data_start + field_len.to_int()
        } else {
          // individual varint
          let (val, new_pos) = read_varint(data, pos)
          ints.push(val.to_int())
          pos = new_pos
        }
      }
      _ => pos = skip_field(data, pos, wire_type)
    }
  }
  OnnxAttribute::{ name, int_value, float_value, ints }
}

///|
/// Parse NodeProto from a length-delimited sub-message.
/// Fields: 1=input(repeated string), 2=output(repeated string), 4=op_type, 5=attribute
fn parse_node_proto(data : Bytes, offset : Int, len : Int) -> OnnxNode {
  let end = offset + len
  let mut pos = offset
  let mut op_type = ""
  let inputs : Array[String] = []
  let outputs : Array[String] = []
  let attributes : Array[OnnxAttribute] = []
  while pos < end {
    let (tag_val, tag_end) = read_varint(data, pos)
    let field_num = (tag_val >> 3).to_int()
    let wire_type = (tag_val & 0x7UL).to_int()
    pos = tag_end
    match field_num {
      1 => {
        // input: string (repeated, each as separate LEN)
        let (field_len, data_start) = read_varint(data, pos)
        inputs.push(parse_pb_string(data, data_start, field_len.to_int()))
        pos = data_start + field_len.to_int()
      }
      2 => {
        // output: string (repeated)
        let (field_len, data_start) = read_varint(data, pos)
        outputs.push(parse_pb_string(data, data_start, field_len.to_int()))
        pos = data_start + field_len.to_int()
      }
      4 => {
        // op_type: string
        let (field_len, data_start) = read_varint(data, pos)
        op_type = parse_pb_string(data, data_start, field_len.to_int())
        pos = data_start + field_len.to_int()
      }
      5 => {
        // attribute: sub-message (repeated)
        let (field_len, data_start) = read_varint(data, pos)
        attributes.push(
          parse_attribute_proto(data, data_start, field_len.to_int()),
        )
        pos = data_start + field_len.to_int()
      }
      _ => pos = skip_field(data, pos, wire_type)
    }
  }
  OnnxNode::{ op_type, inputs, outputs, attributes }
}

///|
/// Extract name from ValueInfoProto (field 1 = name string).
fn parse_value_info_name(data : Bytes, offset : Int, len : Int) -> String {
  let end = offset + len
  let mut pos = offset
  let mut name = ""
  while pos < end {
    let (tag_val, tag_end) = read_varint(data, pos)
    let field_num = (tag_val >> 3).to_int()
    let wire_type = (tag_val & 0x7UL).to_int()
    pos = tag_end
    match field_num {
      1 => {
        // name: string
        let (field_len, data_start) = read_varint(data, pos)
        name = parse_pb_string(data, data_start, field_len.to_int())
        pos = data_start + field_len.to_int()
      }
      _ => pos = skip_field(data, pos, wire_type)
    }
  }
  name
}

///|
/// Parse GraphProto from a length-delimited sub-message.
/// Fields: 1=node, 5=initializer, 11=input, 12=output
fn parse_graph_proto(
  data : Bytes,
  offset : Int,
  len : Int,
) -> OnnxGraph raise OnnxError {
  let end = offset + len
  let mut pos = offset
  let nodes : Array[OnnxNode] = []
  let initializers : Array[OnnxTensor] = []
  let input_names : Array[String] = []
  let output_names : Array[String] = []
  while pos < end {
    let (tag_val, tag_end) = read_varint(data, pos)
    let field_num = (tag_val >> 3).to_int()
    let wire_type = (tag_val & 0x7UL).to_int()
    pos = tag_end
    match field_num {
      1 => {
        // node: sub-message (repeated)
        let (field_len, data_start) = read_varint(data, pos)
        nodes.push(parse_node_proto(data, data_start, field_len.to_int()))
        pos = data_start + field_len.to_int()
      }
      5 => {
        // initializer: TensorProto (repeated)
        let (field_len, data_start) = read_varint(data, pos)
        initializers.push(
          parse_tensor_proto(data, data_start, field_len.to_int()),
        )
        pos = data_start + field_len.to_int()
      }
      11 => {
        // input: ValueInfoProto (repeated)
        let (field_len, data_start) = read_varint(data, pos)
        input_names.push(
          parse_value_info_name(data, data_start, field_len.to_int()),
        )
        pos = data_start + field_len.to_int()
      }
      12 => {
        // output: ValueInfoProto (repeated)
        let (field_len, data_start) = read_varint(data, pos)
        output_names.push(
          parse_value_info_name(data, data_start, field_len.to_int()),
        )
        pos = data_start + field_len.to_int()
      }
      _ => pos = skip_field(data, pos, wire_type)
    }
  }
  OnnxGraph::{ nodes, initializers, input_names, output_names }
}

///|
/// Parse raw ONNX protobuf bytes into an OnnxGraph.
/// Extracts GraphProto from ModelProto (field 7).
pub fn parse_onnx(data : Bytes) -> OnnxGraph raise OnnxError {
  let end = data.length()
  let mut pos = 0
  let mut graph : OnnxGraph? = None
  while pos < end {
    let (tag_val, tag_end) = read_varint(data, pos)
    let field_num = (tag_val >> 3).to_int()
    let wire_type = (tag_val & 0x7UL).to_int()
    pos = tag_end
    match field_num {
      7 => {
        // graph: GraphProto
        let (field_len, data_start) = read_varint(data, pos)
        graph = Some(parse_graph_proto(data, data_start, field_len.to_int()))
        pos = data_start + field_len.to_int()
      }
      _ => pos = skip_field(data, pos, wire_type)
    }
  }
  match graph {
    Some(g) => g
    None => raise InvalidModel("no graph found in ONNX model")
  }
}

///|
/// Find an initializer tensor by name.
fn find_initializer(
  initializers : Array[OnnxTensor],
  name : String,
) -> OnnxTensor? {
  for i = 0; i < initializers.length(); i = i + 1 {
    if initializers[i].name == name {
      return Some(initializers[i])
    }
  }
  None
}

///|
/// Get an integer attribute value by name, with a default.
fn get_int_attr(
  attributes : Array[OnnxAttribute],
  name : String,
  default_ : Int64,
) -> Int64 {
  for i = 0; i < attributes.length(); i = i + 1 {
    if attributes[i].name == name {
      return attributes[i].int_value
    }
  }
  default_
}

///|
/// Get an ints attribute value by name, with a default.
fn get_ints_attr(
  attributes : Array[OnnxAttribute],
  name : String,
  default_ : Array[Int],
) -> Array[Int] {
  for i = 0; i < attributes.length(); i = i + 1 {
    if attributes[i].name == name && attributes[i].ints.length() > 0 {
      return attributes[i].ints
    }
  }
  default_
}

///|
/// Get a float attribute value by name, with a default.
fn get_float_attr(
  attributes : Array[OnnxAttribute],
  name : String,
  default_ : Double,
) -> Double {
  for i = 0; i < attributes.length(); i = i + 1 {
    if attributes[i].name == name {
      return attributes[i].float_value
    }
  }
  default_
}

///|
/// Convert an OnnxTensor to a core Tensor.
fn onnx_tensor_to_tensor(t : OnnxTensor) -> @tensor.Tensor {
  @tensor.Tensor::from_array_with_shape(t.data.copy(), t.shape.copy())
}

///|
/// Load ONNX bytes and convert to MbTorch Linear layers.
/// Interprets Gemm/MatMul+Add as Linear, skips Relu nodes.
pub fn load_onnx_model(
  data : Bytes,
) -> Array[(String, @nn.Linear)] raise OnnxError {
  let graph = parse_onnx(data)
  let result : Array[(String, @nn.Linear)] = []
  let mut skip_next_add = false
  let mut i = 0
  while i < graph.nodes.length() {
    let node = graph.nodes[i]
    match node.op_type {
      "Gemm" => {
        // inputs: [X, W, B]
        guard node.inputs.length() >= 2 else {
          raise InvalidModel("Gemm node needs at least 2 inputs")
        }
        let weight_name = node.inputs[1]
        let weight_tensor = match
          find_initializer(graph.initializers, weight_name) {
          Some(t) => t
          None =>
            raise InvalidModel("weight initializer not found: " + weight_name)
        }
        // Validate dtype: 1=FLOAT, 10=FLOAT16
        guard weight_tensor.dtype == 1 || weight_tensor.dtype == 10 else {
          raise UnsupportedDtype(
            "tensor " +
            weight_name +
            ": expected FLOAT (1) or FLOAT16 (10), got " +
            weight_tensor.dtype.to_string(),
          )
        }
        let mut weight = onnx_tensor_to_tensor(weight_tensor)
        // Check transB attribute
        let trans_b = get_int_attr(node.attributes, "transB", 0L)
        if trans_b == 1L {
          weight = weight.transpose()
        }
        // Get bias
        let bias = if node.inputs.length() >= 3 {
          let bias_name = node.inputs[2]
          match find_initializer(graph.initializers, bias_name) {
            Some(t) => onnx_tensor_to_tensor(t)
            None =>
              raise InvalidModel("bias initializer not found: " + bias_name)
          }
        } else {
          // No bias: create zeros
          let out_features = weight.shape()[1]
          @tensor.Tensor::zeros([out_features])
        }
        let output_name = if node.outputs.length() > 0 {
          node.outputs[0]
        } else {
          "layer_" + i.to_string()
        }
        let linear = @nn.Linear::from_tensors(weight, bias)
        result.push((output_name, linear))
      }
      "MatMul" => {
        // inputs: [X, W]
        guard node.inputs.length() >= 2 else {
          raise InvalidModel("MatMul node needs at least 2 inputs")
        }
        let weight_name = node.inputs[1]
        let weight_tensor = match
          find_initializer(graph.initializers, weight_name) {
          Some(t) => t
          None =>
            raise InvalidModel("weight initializer not found: " + weight_name)
        }
        guard weight_tensor.dtype == 1 || weight_tensor.dtype == 10 else {
          raise UnsupportedDtype(
            "tensor " +
            weight_name +
            ": expected FLOAT (1) or FLOAT16 (10), got " +
            weight_tensor.dtype.to_string(),
          )
        }
        let weight = onnx_tensor_to_tensor(weight_tensor)
        let matmul_output = if node.outputs.length() > 0 {
          node.outputs[0]
        } else {
          ""
        }
        // Check if next node is Add that uses this output
        let mut bias = @tensor.Tensor::zeros([weight.shape()[1]])
        let mut output_name = matmul_output
        if i + 1 < graph.nodes.length() {
          let next_node = graph.nodes[i + 1]
          if next_node.op_type == "Add" && next_node.inputs.length() >= 2 {
            // Check if one of the Add inputs is our MatMul output
            let mut fused = false
            for j = 0; j < next_node.inputs.length(); j = j + 1 {
              if next_node.inputs[j] == matmul_output {
                // Find the other input as bias
                let bias_idx = if j == 0 { 1 } else { 0 }
                let bias_name = next_node.inputs[bias_idx]
                match find_initializer(graph.initializers, bias_name) {
                  Some(t) => {
                    bias = onnx_tensor_to_tensor(t)
                    fused = true
                    skip_next_add = true
                    if next_node.outputs.length() > 0 {
                      output_name = next_node.outputs[0]
                    }
                  }
                  None => ()
                }
                if fused {
                  break
                }
              }
            }
          }
        }
        let linear = @nn.Linear::from_tensors(weight, bias)
        result.push((output_name, linear))
      }
      "Add" =>
        if skip_next_add {
          skip_next_add = false
        } else {
          raise UnsupportedOp("standalone Add is not supported in Phase 1")
        }
      "Relu" =>
        // Skip: activation is applied externally
        ()
      _ => raise UnsupportedOp("unsupported op: " + node.op_type)
    }
    i = i + 1
  }
  result
}
