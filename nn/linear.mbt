///|
/// Linear layer: y = x @ weight + bias
/// weight shape: [in_features, out_features]
/// bias shape:   [out_features] (1D, expanded to [batch, out] in forward)
pub struct Linear {
  weight : @autograd.Variable
  bias : @autograd.Variable
}

///|
pub fn Linear::new(
  in_features : Int,
  out_features : Int,
  seed? : UInt64 = 42UL,
) -> Linear {
  let weight_data = @tensor.Tensor::rand([in_features, out_features], seed~)
  let bias_data = @tensor.Tensor::zeros([out_features])
  let weight = @autograd.Variable::new(weight_data, requires_grad=true)
  let bias = @autograd.Variable::new(bias_data, requires_grad=true)
  { weight, bias }
}

///|
/// input shape: [batch, in_features]
/// output shape: [batch, out_features]
pub fn Linear::forward(
  self : Linear,
  input : @autograd.Variable,
) -> @autograd.Variable {
  // x @ weight: [batch, in] @ [in, out] = [batch, out]
  let xw = input.matmul(self.weight)
  // Expand bias [out] to [batch, out] via autograd-tracked op
  let batch = input.shape()[0]
  let bias_expanded = self.bias.expand_rows(batch)
  xw.add(bias_expanded)
}

///|
pub fn Linear::parameters(self : Linear) -> Array[@autograd.Variable] {
  [self.weight, self.bias]
}
