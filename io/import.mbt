///|
/// Load a model from ONNX (structure) + safetensors (weights).
/// ONNX graph defines the network topology (Gemm/MatMul/Add/Relu nodes),
/// while safetensors provides the weight tensors by name.
/// Returns an array of (output_name, Linear) pairs.
pub fn load_model_from_onnx_and_safetensors(
  onnx_data : Bytes,
  safetensors_data : Bytes,
) -> Array[(String, @nn.Linear)] raise OnnxError {
  // 1. Parse ONNX graph structure
  let graph = parse_onnx(onnx_data)
  // 2. Load safetensors weights into a name→Tensor map
  let tensors = load_safetensors(safetensors_data) catch {
    _ => raise InvalidModel("failed to parse safetensors data")
  }
  let weight_map : Map[String, @tensor.Tensor] = {}
  for i = 0; i < tensors.length(); i = i + 1 {
    let (name, tensor) = tensors[i]
    weight_map.set(name, tensor)
  }
  // Helper: look up a tensor by name from the safetensors map
  fn lookup(
    map : Map[String, @tensor.Tensor],
    name : String,
  ) -> @tensor.Tensor raise OnnxError {
    match map.get(name) {
      Some(t) => t
      None => raise InvalidModel("tensor not found in safetensors: " + name)
    }
  }

  // 3. Walk ONNX nodes and build Linear layers using safetensors weights
  let result : Array[(String, @nn.Linear)] = []
  let mut skip_next_add = false
  let mut i = 0
  while i < graph.nodes.length() {
    let node = graph.nodes[i]
    match node.op_type {
      "Gemm" => {
        guard node.inputs.length() >= 2 else {
          raise InvalidModel("Gemm node needs at least 2 inputs")
        }
        let mut weight = lookup(weight_map, node.inputs[1])
        let trans_b = get_int_attr(node.attributes, "transB", 0L)
        if trans_b == 1L {
          weight = weight.transpose()
        }
        let bias = if node.inputs.length() >= 3 {
          lookup(weight_map, node.inputs[2])
        } else {
          @tensor.Tensor::zeros([weight.shape()[1]])
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
        guard node.inputs.length() >= 2 else {
          raise InvalidModel("MatMul node needs at least 2 inputs")
        }
        let weight = lookup(weight_map, node.inputs[1])
        let matmul_output = if node.outputs.length() > 0 {
          node.outputs[0]
        } else {
          ""
        }
        let mut bias = @tensor.Tensor::zeros([weight.shape()[1]])
        let mut output_name = matmul_output
        if i + 1 < graph.nodes.length() {
          let next_node = graph.nodes[i + 1]
          if next_node.op_type == "Add" && next_node.inputs.length() >= 2 {
            let mut fused = false
            for j = 0; j < next_node.inputs.length(); j = j + 1 {
              if next_node.inputs[j] == matmul_output {
                let bias_idx = if j == 0 { 1 } else { 0 }
                let bias_name = next_node.inputs[bias_idx]
                match weight_map.get(bias_name) {
                  Some(t) => {
                    bias = t
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
      "Relu" => ()
      _ => raise UnsupportedOp("unsupported op: " + node.op_type)
    }
    i = i + 1
  }
  result
}

///|
/// Load a CNN model from ONNX (structure) + safetensors (weights).
/// Supports Conv, BatchNormalization, Relu, Flatten, Gemm/MatMul+Add nodes.
/// Returns an array of (name, Layer) pairs.
pub fn load_cnn_model_from_onnx_and_safetensors(
  onnx_data : Bytes,
  safetensors_data : Bytes,
) -> Array[(String, @nn.Layer)] raise OnnxError {
  let graph = parse_onnx(onnx_data)
  let tensors = load_safetensors(safetensors_data) catch {
    _ => raise InvalidModel("failed to parse safetensors data")
  }
  let weight_map : Map[String, @tensor.Tensor] = {}
  for i = 0; i < tensors.length(); i = i + 1 {
    let (name, tensor) = tensors[i]
    weight_map.set(name, tensor)
  }
  fn lookup(
    map : Map[String, @tensor.Tensor],
    name : String,
  ) -> @tensor.Tensor raise OnnxError {
    match map.get(name) {
      Some(t) => t
      None => raise InvalidModel("tensor not found in safetensors: " + name)
    }
  }

  let result : Array[(String, @nn.Layer)] = []
  let mut skip_next_add = false
  let mut i = 0
  while i < graph.nodes.length() {
    let node = graph.nodes[i]
    let output_name = if node.outputs.length() > 0 {
      node.outputs[0]
    } else {
      "layer_" + i.to_string()
    }
    match node.op_type {
      "Conv" => {
        guard node.inputs.length() >= 2 else {
          raise InvalidModel("Conv node needs at least 2 inputs")
        }
        // Validate groups and dilations
        let group = get_int_attr(node.attributes, "group", 1L)
        guard group == 1L else {
          raise UnsupportedOp(
            "Conv with group=" + group.to_int().to_string() + " not supported (only group=1)",
          )
        }
        let dilations = get_ints_attr(node.attributes, "dilations", [1, 1])
        guard dilations.length() <= 2 &&
          (dilations.length() == 0 ||
          (dilations[0] == 1 &&
          (dilations.length() < 2 || dilations[1] == 1))) else {
          raise UnsupportedOp("Conv with dilations != [1,1] not supported")
        }
        let weight = lookup(weight_map, node.inputs[1])
        let bias : @tensor.Tensor? = if node.inputs.length() >= 3 &&
          node.inputs[2] != "" {
          Some(lookup(weight_map, node.inputs[2]))
        } else {
          None
        }
        let strides = get_ints_attr(node.attributes, "strides", [1, 1])
        let pads = get_ints_attr(node.attributes, "pads", [0, 0, 0, 0])
        let conv = @nn.Conv2d::from_tensors(
          weight,
          bias_data=bias,
          stride=strides,
          padding=pads,
        )
        result.push((output_name, @nn.Layer::Conv2d(conv)))
      }
      "BatchNormalization" => {
        guard node.inputs.length() >= 5 else {
          raise InvalidModel("BatchNormalization needs 5 inputs")
        }
        let gamma = lookup(weight_map, node.inputs[1])
        let beta = lookup(weight_map, node.inputs[2])
        let mean = lookup(weight_map, node.inputs[3])
        let var_ = lookup(weight_map, node.inputs[4])
        let eps = get_float_attr(node.attributes, "epsilon", 1.0e-5)
        let bn = @nn.BatchNorm2d::from_tensors(
          gamma=gamma,
          beta=beta,
          mean=mean,
          var_=var_,
          eps=eps,
        )
        result.push((output_name, @nn.Layer::BatchNorm2d(bn)))
      }
      "Relu" => result.push((output_name, @nn.Layer::Relu))
      "Flatten" => result.push((output_name, @nn.Layer::Flatten))
      "Gemm" => {
        guard node.inputs.length() >= 2 else {
          raise InvalidModel("Gemm node needs at least 2 inputs")
        }
        let mut weight = lookup(weight_map, node.inputs[1])
        let trans_b = get_int_attr(node.attributes, "transB", 0L)
        if trans_b == 1L {
          weight = weight.transpose()
        }
        let bias = if node.inputs.length() >= 3 {
          lookup(weight_map, node.inputs[2])
        } else {
          @tensor.Tensor::zeros([weight.shape()[1]])
        }
        let linear = @nn.Linear::from_tensors(weight, bias)
        result.push((output_name, @nn.Layer::Linear(linear)))
      }
      "MatMul" => {
        guard node.inputs.length() >= 2 else {
          raise InvalidModel("MatMul node needs at least 2 inputs")
        }
        let weight = lookup(weight_map, node.inputs[1])
        let matmul_output = output_name
        let mut bias = @tensor.Tensor::zeros([weight.shape()[1]])
        let mut final_name = matmul_output
        if i + 1 < graph.nodes.length() {
          let next_node = graph.nodes[i + 1]
          if next_node.op_type == "Add" && next_node.inputs.length() >= 2 {
            let mut fused = false
            for j = 0; j < next_node.inputs.length(); j = j + 1 {
              if next_node.inputs[j] == matmul_output {
                let bias_idx = if j == 0 { 1 } else { 0 }
                let bias_name = next_node.inputs[bias_idx]
                match weight_map.get(bias_name) {
                  Some(t) => {
                    bias = t
                    fused = true
                    skip_next_add = true
                    if next_node.outputs.length() > 0 {
                      final_name = next_node.outputs[0]
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
        result.push((final_name, @nn.Layer::Linear(linear)))
      }
      "Add" =>
        if skip_next_add {
          skip_next_add = false
        } else {
          raise UnsupportedOp("standalone Add is not supported")
        }
      _ => raise UnsupportedOp("unsupported op: " + node.op_type)
    }
    i = i + 1
  }
  result
}

///|
/// Import a SelfAttention layer from ONNX bytes.
/// Expects the pattern: 3×Gemm(Q/K/V) + Transpose + MatMul + Div + Softmax + MatMul + Gemm(O)
pub fn import_self_attention_from_onnx(
  onnx_data : Bytes,
  num_heads~ : Int = 1,
) -> @nn.SelfAttention raise OnnxError {
  let graph = parse_onnx(onnx_data)
  let nodes = graph.nodes
  guard nodes.length() == 9 else {
    raise InvalidModel(
      "self-attention import expects 9 nodes, got \{nodes.length()}",
    )
  }
  // Validate pattern
  guard nodes[0].op_type == "Gemm" &&
    nodes[1].op_type == "Gemm" &&
    nodes[2].op_type == "Gemm" &&
    nodes[3].op_type == "Transpose" &&
    nodes[4].op_type == "MatMul" &&
    nodes[5].op_type == "Div" &&
    nodes[6].op_type == "Softmax" &&
    nodes[7].op_type == "MatMul" &&
    nodes[8].op_type == "Gemm" else {
    raise UnsupportedOp(
      "self-attention pattern mismatch: expected Gemm×3+Transpose+MatMul+Div+Softmax+MatMul+Gemm",
    )
  }
  // Validate Q/K/V share same input (self-attention)
  guard nodes[0].inputs[0] == nodes[1].inputs[0] &&
    nodes[0].inputs[0] == nodes[2].inputs[0] else {
    raise UnsupportedOp(
      "not self-attention: Q/K/V projections have different inputs",
    )
  }
  // Build initializer map
  let init_map : Map[String, OnnxTensor] = {}
  for i = 0; i < graph.initializers.length(); i = i + 1 {
    init_map.set(graph.initializers[i].name, graph.initializers[i])
  }
  fn load_gemm_as_linear(
    node : OnnxNode,
    init_map : Map[String, OnnxTensor],
  ) -> @nn.Linear raise OnnxError {
    guard node.inputs.length() >= 3 else {
      raise InvalidModel("Gemm needs at least 3 inputs for self-attention")
    }
    let w_init = match init_map.get(node.inputs[1]) {
      Some(t) => t
      None => raise InvalidModel("weight not found: " + node.inputs[1])
    }
    let b_init = match init_map.get(node.inputs[2]) {
      Some(t) => t
      None => raise InvalidModel("bias not found: " + node.inputs[2])
    }
    let mut weight = @tensor.Tensor::from_array_with_shape(
      w_init.data,
      w_init.shape,
    )
    let trans_b = get_int_attr(node.attributes, "transB", 0L)
    if trans_b == 1L {
      weight = weight.transpose()
    }
    let bias = @tensor.Tensor::from_array_with_shape(b_init.data, b_init.shape)
    @nn.Linear::from_tensors(weight, bias)
  }

  let wq = load_gemm_as_linear(nodes[0], init_map)
  let wk = load_gemm_as_linear(nodes[1], init_map)
  let wv = load_gemm_as_linear(nodes[2], init_map)
  let wo = load_gemm_as_linear(nodes[8], init_map)
  @nn.SelfAttention::from_linears(wq, wk, wv, wo, num_heads)
}
