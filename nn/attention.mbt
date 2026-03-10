///|
/// Scaled Dot-Product Self-Attention layer.
/// Supports single-head and multi-head attention.
/// Input: [batch, seq_len, d_model] or [seq_len, d_model] (batch=1)
/// Output: same shape as input
pub struct SelfAttention {
  wq : Linear
  wk : Linear
  wv : Linear
  wo : Linear
  num_heads : Int
  d_k : Int // = d_model / num_heads
}

///|
pub fn SelfAttention::new(
  d_model : Int,
  num_heads~ : Int = 1,
  seed~ : UInt64 = 42UL,
) -> SelfAttention {
  guard d_model % num_heads == 0 else {
    abort(
      "SelfAttention::new: d_model \{d_model} must be divisible by num_heads \{num_heads}",
    )
  }
  let d_k = d_model / num_heads
  let wq = Linear::new(d_model, d_model, seed~)
  let wk = Linear::new(d_model, d_model, seed=(seed + 1UL))
  let wv = Linear::new(d_model, d_model, seed=(seed + 2UL))
  let wo = Linear::new(d_model, d_model, seed=(seed + 3UL))
  { wq, wk, wv, wo, num_heads, d_k }
}

///|
/// Create SelfAttention from pre-existing Linear layers (for ONNX import).
pub fn SelfAttention::from_linears(
  wq : Linear,
  wk : Linear,
  wv : Linear,
  wo : Linear,
  num_heads : Int,
) -> SelfAttention {
  let d_model = wq.weight.shape()[0]
  guard d_model % num_heads == 0 else {
    abort(
      "SelfAttention::from_linears: d_model \{d_model} not divisible by num_heads \{num_heads}",
    )
  }
  let d_k = d_model / num_heads
  { wq, wk, wv, wo, num_heads, d_k }
}

///|
pub fn SelfAttention::forward(
  self : SelfAttention,
  input : @autograd.Variable,
) -> @autograd.Variable {
  let shape = input.shape()
  // Handle 2D input [seq_len, d_model] by adding batch dim
  let is_2d = shape.length() == 2
  let x = if is_2d {
    input.reshape([1, shape[0], shape[1]])
  } else {
    input
  }
  let batch = x.shape()[0]
  let seq_len = x.shape()[1]
  let d_model = x.shape()[2]
  let num_heads = self.num_heads
  let d_k = self.d_k
  // Flatten batch*seq for linear projections: [B*S, d_model]
  let x_flat = x.reshape([batch * seq_len, d_model])
  // Q, K, V projections
  let q = self.wq.forward(x_flat) // [B*S, d_model]
  let k = self.wk.forward(x_flat)
  let v = self.wv.forward(x_flat)
  // Reshape for multi-head: [B*H, S, d_k]
  let q3 = q.reshape([batch * num_heads, seq_len, d_k])
  let k3 = k.reshape([batch * num_heads, seq_len, d_k])
  let v3 = v.reshape([batch * num_heads, seq_len, d_k])
  // Scaled dot-product attention
  // scores = Q @ K^T / sqrt(d_k)  -> [B*H, S, S]
  let scores = q3
    .batched_matmul_2d(k3.transpose_last2())
    .scale(1.0 / d_k.to_double().sqrt())
  // attn_weights = softmax(scores) -> [B*H, S, S]
  let attn = scores.softmax()
  // context = attn @ V -> [B*H, S, d_k]
  let context = attn.batched_matmul_2d(v3)
  // Reshape back: [B*S, d_model]
  let context_flat = context.reshape([batch * seq_len, d_model])
  // Output projection
  let output_flat = self.wo.forward(context_flat) // [B*S, d_model]
  // Reshape to [B, S, d_model]
  let output = output_flat.reshape([batch, seq_len, d_model])
  if is_2d {
    output.reshape([seq_len, d_model])
  } else {
    output
  }
}

///|
pub fn SelfAttention::parameters(
  self : SelfAttention,
) -> Array[@autograd.Variable] {
  let params : Array[@autograd.Variable] = []
  let wq_p = self.wq.parameters()
  let wk_p = self.wk.parameters()
  let wv_p = self.wv.parameters()
  let wo_p = self.wo.parameters()
  for i = 0; i < wq_p.length(); i = i + 1 {
    params.push(wq_p[i])
  }
  for i = 0; i < wk_p.length(); i = i + 1 {
    params.push(wk_p[i])
  }
  for i = 0; i < wv_p.length(); i = i + 1 {
    params.push(wv_p[i])
  }
  for i = 0; i < wo_p.length(); i = i + 1 {
    params.push(wo_p[i])
  }
  params
}
