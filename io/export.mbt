// ONNX export: MbTorch sequential models → ONNX protobuf bytes

///|
/// Convert an array of MbTorch layers to ONNX protobuf bytes.
/// layers: Sequential model as Array[@nn.Layer]
/// input_shape: Shape of network input (e.g. [1, 784] for MLP, [1, 1, 28, 28] for CNN)
/// Returns: Valid ONNX ModelProto bytes (opset 13) loadable by onnxruntime
pub fn export_onnx(
  layers : Array[@nn.Layer],
  input_shape : Array[Int],
) -> Bytes {
  let graph = layers_to_onnx_graph(layers, input_shape)
  let graph_bytes = encode_onnx_graph(graph)
  build_model_proto_with_opset(graph_bytes, 13)
}

///|
/// Build an OnnxGraph from a sequential layer array and input shape.
fn layers_to_onnx_graph(
  layers : Array[@nn.Layer],
  _input_shape : Array[Int],
) -> OnnxGraph {
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
      @nn.Layer::SelfAttention(sa) => {
        let (sa_nodes, sa_inits) = export_self_attention(
          sa, i, current_name, out_name,
        )
        for j = 0; j < sa_nodes.length(); j = j + 1 {
          nodes.push(sa_nodes[j])
        }
        for j = 0; j < sa_inits.length(); j = j + 1 {
          initializers.push(sa_inits[j])
        }
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
  encode_onnx_graph_with_dtype(graph, @tensor.Float32)
}

///|
/// Encode an OnnxGraph to protobuf bytes with specified dtype for initializers.
fn encode_onnx_graph_with_dtype(
  graph : OnnxGraph,
  dtype : @tensor.DType,
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
      @tensor.Float16 =>
        init_bytes.push(build_tensor_proto_f16(init.name, init.shape, init.data))
      @tensor.Float32 | @tensor.Float64 =>
        init_bytes.push(build_tensor_proto(init.name, init.shape, init.data))
      @tensor.Int8 =>
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
/// Export a SelfAttention layer as a subgraph:
/// Gemm(Q) + Gemm(K) + Gemm(V) + Transpose(K^T) + MatMul(QK^T) + Div(scale) + Softmax + MatMul(attn@V) + Gemm(O)
fn export_self_attention(
  sa : @nn.SelfAttention,
  layer_idx : Int,
  input_name : String,
  output_name : String,
) -> (Array[OnnxNode], Array[OnnxTensor]) {
  let prefix = "sa_\{layer_idx}"
  let nodes : Array[OnnxNode] = []
  let inits : Array[OnnxTensor] = []
  // Helper: export a Linear as Gemm (transB=1)
  fn add_gemm(
    linear : @nn.Linear,
    name : String,
    in_name : String,
    out_name : String,
    nodes : Array[OnnxNode],
    inits : Array[OnnxTensor],
  ) {
    let w_name = name + "_weight"
    let b_name = name + "_bias"
    let w_tensor = linear.weight.data().transpose()
    let b_tensor = linear.bias.data()
    inits.push(
      OnnxTensor::{
        name: w_name,
        dtype: 1,
        shape: w_tensor.shape(),
        data: w_tensor.to_array(),
      },
    )
    inits.push(
      OnnxTensor::{
        name: b_name,
        dtype: 1,
        shape: b_tensor.shape(),
        data: b_tensor.to_array(),
      },
    )
    nodes.push(
      OnnxNode::{
        op_type: "Gemm",
        inputs: [in_name, w_name, b_name],
        outputs: [out_name],
        attributes: [
          OnnxAttribute::{
            name: "transB",
            int_value: 1L,
            float_value: 0.0,
            ints: [],
          },
        ],
      },
    )
  }

  // 1. Q, K, V projections (Gemm)
  let q_name = prefix + "_q"
  let k_name = prefix + "_k"
  let v_name = prefix + "_v"
  add_gemm(sa.wq, prefix + "_wq", input_name, q_name, nodes, inits)
  add_gemm(sa.wk, prefix + "_wk", input_name, k_name, nodes, inits)
  add_gemm(sa.wv, prefix + "_wv", input_name, v_name, nodes, inits)
  // 2. Transpose K -> K^T
  let kt_name = prefix + "_kt"
  nodes.push(
    OnnxNode::{
      op_type: "Transpose",
      inputs: [k_name],
      outputs: [kt_name],
      attributes: [
        OnnxAttribute::{
          name: "perm",
          int_value: 0L,
          float_value: 0.0,
          ints: [1, 0],
        },
      ],
    },
  )
  // 3. MatMul(Q, K^T) -> scores
  let scores_name = prefix + "_scores"
  nodes.push(
    OnnxNode::{
      op_type: "MatMul",
      inputs: [q_name, kt_name],
      outputs: [scores_name],
      attributes: [],
    },
  )
  // 4. Div(scores, sqrt(d_k)) -> scaled_scores
  let scale_name = prefix + "_scale"
  let d_k = sa.d_k.to_double()
  inits.push(
    OnnxTensor::{
      name: scale_name,
      dtype: 1,
      shape: [1],
      data: [d_k.sqrt()],
    },
  )
  let scaled_name = prefix + "_scaled"
  nodes.push(
    OnnxNode::{
      op_type: "Div",
      inputs: [scores_name, scale_name],
      outputs: [scaled_name],
      attributes: [],
    },
  )
  // 5. Softmax -> attn_weights
  let attn_name = prefix + "_attn"
  nodes.push(
    OnnxNode::{
      op_type: "Softmax",
      inputs: [scaled_name],
      outputs: [attn_name],
      attributes: [
        OnnxAttribute::{
          name: "axis",
          int_value: -1L,
          float_value: 0.0,
          ints: [],
        },
      ],
    },
  )
  // 6. MatMul(attn, V) -> context
  let context_name = prefix + "_context"
  nodes.push(
    OnnxNode::{
      op_type: "MatMul",
      inputs: [attn_name, v_name],
      outputs: [context_name],
      attributes: [],
    },
  )
  // 7. Output projection (Gemm)
  add_gemm(sa.wo, prefix + "_wo", context_name, output_name, nodes, inits)
  (nodes, inits)
}

///|
/// Export MbTorch layers to ONNX protobuf bytes with specified dtype for weights.
pub fn export_onnx_with_dtype(
  layers : Array[@nn.Layer],
  input_shape : Array[Int],
  dtype : @tensor.DType,
) -> Bytes {
  let graph = layers_to_onnx_graph(layers, input_shape)
  let graph_bytes = encode_onnx_graph_with_dtype(graph, dtype)
  build_model_proto_with_opset(graph_bytes, 13)
}
