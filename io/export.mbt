// ONNX export: MbTorch sequential models → ONNX protobuf bytes

///|
/// Convert an array of MbTorch layers to ONNX protobuf bytes.
/// layers: Sequential model as Array[@nn.Layer]
/// input_shape: Shape of network input (e.g. [1, 784] for MLP, [1, 1, 28, 28] for CNN)
/// Returns: Valid ONNX ModelProto bytes (opset 13) loadable by onnxruntime
pub fn export_onnx(
  layers : Array[@nn.Layer],
  input_shape : Array[Int],
) -> Bytes raise OnnxError {
  let graph = layers_to_onnx_graph(layers, input_shape)
  let graph_bytes = encode_onnx_graph(graph)
  build_model_proto_with_opset(graph_bytes, 13)
}

///|
/// Build an OnnxGraph from a sequential layer array and input shape.
fn layers_to_onnx_graph(
  layers : Array[@nn.Layer],
  _input_shape : Array[Int],
) -> OnnxGraph raise OnnxError {
  let nodes : Array[OnnxNode] = []
  let initializers : Array[OnnxTensor] = []
  let mut current_name = "input"
  let mut tensor_idx = 0
  for i = 0; i < layers.length(); i = i + 1 {
    let is_last = i == layers.length() - 1
    let out_name = if is_last { "output" } else { "t_\{tensor_idx}" }
    tensor_idx = tensor_idx + 1
    match layers[i] {
      @nn.Layer::Linear(l) => {
        let (node, inits) = export_linear(l, i, current_name, out_name)
        nodes.push(node)
        for j = 0; j < inits.length(); j = j + 1 {
          initializers.push(inits[j])
        }
      }
      @nn.Layer::Conv2d(c) => {
        let (node, inits) = export_conv2d(c, i, current_name, out_name)
        nodes.push(node)
        for j = 0; j < inits.length(); j = j + 1 {
          initializers.push(inits[j])
        }
      }
      @nn.Layer::BatchNorm2d(bn) => {
        let (node, inits) = export_batchnorm2d(bn, i, current_name, out_name)
        nodes.push(node)
        for j = 0; j < inits.length(); j = j + 1 {
          initializers.push(inits[j])
        }
      }
      @nn.Layer::Relu => {
        let node = OnnxNode::{
          op_type: "Relu",
          inputs: [current_name],
          outputs: [out_name],
          attributes: [],
        }
        nodes.push(node)
      }
      @nn.Layer::Flatten => {
        let node = OnnxNode::{
          op_type: "Flatten",
          inputs: [current_name],
          outputs: [out_name],
          attributes: [
            OnnxAttribute::{
              name: "axis",
              int_value: 1L,
              float_value: 0.0,
              ints: [],
            },
          ],
        }
        nodes.push(node)
      }
    }
    current_name = out_name
  }
  OnnxGraph::{
    nodes,
    initializers,
    input_names: ["input"],
    output_names: ["output"],
  }
}

///|
/// Export a Linear layer as a Gemm node (transB=1).
/// MbTorch weight: [in, out] → ONNX weight: [out, in] (transposed)
fn export_linear(
  l : @nn.Linear,
  layer_idx : Int,
  input_name : String,
  output_name : String,
) -> (OnnxNode, Array[OnnxTensor]) {
  let weight_name = "layer_\{layer_idx}_weight"
  let bias_name = "layer_\{layer_idx}_bias"
  // Transpose weight from [in, out] to [out, in]
  let w_tensor = l.weight.data().transpose()
  let w_shape = w_tensor.shape()
  let w_data = w_tensor.to_array()
  let b_tensor = l.bias.data()
  let b_shape = b_tensor.shape()
  let b_data = b_tensor.to_array()
  let weight_init = OnnxTensor::{
    name: weight_name,
    dtype: 1,
    shape: w_shape,
    data: w_data,
  }
  let bias_init = OnnxTensor::{
    name: bias_name,
    dtype: 1,
    shape: b_shape,
    data: b_data,
  }
  let node = OnnxNode::{
    op_type: "Gemm",
    inputs: [input_name, weight_name, bias_name],
    outputs: [output_name],
    attributes: [
      OnnxAttribute::{
        name: "transB",
        int_value: 1L,
        float_value: 0.0,
        ints: [],
      },
    ],
  }
  (node, [weight_init, bias_init])
}

///|
/// Export a Conv2d layer as a Conv node.
fn export_conv2d(
  c : @nn.Conv2d,
  layer_idx : Int,
  input_name : String,
  output_name : String,
) -> (OnnxNode, Array[OnnxTensor]) {
  let weight_name = "layer_\{layer_idx}_weight"
  let w_tensor = c.weight.data()
  let weight_init = OnnxTensor::{
    name: weight_name,
    dtype: 1,
    shape: w_tensor.shape(),
    data: w_tensor.to_array(),
  }
  let inits : Array[OnnxTensor] = [weight_init]
  let inputs : Array[String] = [input_name, weight_name]
  // Optional bias
  match c.bias {
    Some(bias_var) => {
      let bias_name = "layer_\{layer_idx}_bias"
      let b_tensor = bias_var.data()
      inits.push(
        OnnxTensor::{
          name: bias_name,
          dtype: 1,
          shape: b_tensor.shape(),
          data: b_tensor.to_array(),
        },
      )
      inputs.push(bias_name)
    }
    None => ()
  }
  let attrs : Array[OnnxAttribute] = [
    OnnxAttribute::{
      name: "strides",
      int_value: 0L,
      float_value: 0.0,
      ints: c.stride,
    },
    OnnxAttribute::{
      name: "pads",
      int_value: 0L,
      float_value: 0.0,
      ints: c.padding,
    },
    OnnxAttribute::{
      name: "dilations",
      int_value: 0L,
      float_value: 0.0,
      ints: [1, 1],
    },
    OnnxAttribute::{
      name: "group",
      int_value: 1L,
      float_value: 0.0,
      ints: [],
    },
  ]
  let node = OnnxNode::{
    op_type: "Conv",
    inputs,
    outputs: [output_name],
    attributes: attrs,
  }
  (node, inits)
}

///|
/// Export a BatchNorm2d layer as a BatchNormalization node.
fn export_batchnorm2d(
  bn : @nn.BatchNorm2d,
  layer_idx : Int,
  input_name : String,
  output_name : String,
) -> (OnnxNode, Array[OnnxTensor]) {
  let gamma_name = "layer_\{layer_idx}_gamma"
  let beta_name = "layer_\{layer_idx}_beta"
  let mean_name = "layer_\{layer_idx}_running_mean"
  let var_name = "layer_\{layer_idx}_running_var"
  let gamma_init = OnnxTensor::{
    name: gamma_name,
    dtype: 1,
    shape: bn.gamma.shape(),
    data: bn.gamma.to_array(),
  }
  let beta_init = OnnxTensor::{
    name: beta_name,
    dtype: 1,
    shape: bn.beta.shape(),
    data: bn.beta.to_array(),
  }
  let mean_init = OnnxTensor::{
    name: mean_name,
    dtype: 1,
    shape: bn.running_mean.shape(),
    data: bn.running_mean.to_array(),
  }
  let var_init = OnnxTensor::{
    name: var_name,
    dtype: 1,
    shape: bn.running_var.shape(),
    data: bn.running_var.to_array(),
  }
  let node = OnnxNode::{
    op_type: "BatchNormalization",
    inputs: [input_name, gamma_name, beta_name, mean_name, var_name],
    outputs: [output_name],
    attributes: [
      OnnxAttribute::{
        name: "epsilon",
        int_value: 0L,
        float_value: bn.eps,
        ints: [],
      },
    ],
  }
  (node, [gamma_init, beta_init, mean_init, var_init])
}

///|
/// Encode an OnnxGraph to protobuf bytes (GraphProto).
fn encode_onnx_graph(graph : OnnxGraph) -> Array[Byte] {
  encode_onnx_graph_with_dtype(graph, "F32")
}

///|
/// Encode an OnnxGraph to protobuf bytes with specified dtype for initializers.
fn encode_onnx_graph_with_dtype(
  graph : OnnxGraph,
  dtype : String,
) -> Array[Byte] {
  // Encode each node
  let node_bytes : Array[Array[Byte]] = []
  for i = 0; i < graph.nodes.length(); i = i + 1 {
    let node = graph.nodes[i]
    if node.attributes.length() == 0 {
      node_bytes.push(
        build_node_proto(node.op_type, node.inputs, node.outputs, []),
      )
    } else {
      node_bytes.push(
        build_node_proto_full(
          node.op_type,
          node.inputs,
          node.outputs,
          node.attributes,
        ),
      )
    }
  }
  // Encode each initializer
  let init_bytes : Array[Array[Byte]] = []
  for i = 0; i < graph.initializers.length(); i = i + 1 {
    let init = graph.initializers[i]
    match dtype {
      "F16" =>
        init_bytes.push(build_tensor_proto_f16(init.name, init.shape, init.data))
      _ =>
        init_bytes.push(build_tensor_proto(init.name, init.shape, init.data))
    }
  }
  build_graph_proto(
    node_bytes,
    init_bytes,
    graph.input_names,
    graph.output_names,
  )
}

///|
/// Export MbTorch layers to ONNX protobuf bytes with specified dtype for weights.
/// dtype: "F32" (default) or "F16" (half-precision weights).
pub fn export_onnx_with_dtype(
  layers : Array[@nn.Layer],
  input_shape : Array[Int],
  dtype : String,
) -> Bytes raise OnnxError {
  let graph = layers_to_onnx_graph(layers, input_shape)
  let graph_bytes = encode_onnx_graph_with_dtype(graph, dtype)
  build_model_proto_with_opset(graph_bytes, 13)
}
