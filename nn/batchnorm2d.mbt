///|
/// BatchNorm2d layer for inference only (no backward, no running stats update).
/// Applies: y = (x - mean) / sqrt(var + eps) * gamma + beta
/// All parameters are per-channel (shape: [C]).
pub struct BatchNorm2d {
  gamma : @tensor.Tensor
  beta : @tensor.Tensor
  running_mean : @tensor.Tensor
  running_var : @tensor.Tensor
  eps : Double
}

///|
/// Create a BatchNorm2d from pre-existing tensors.
/// All tensors must be 1D with size = number of channels.
pub fn BatchNorm2d::from_tensors(
  gamma~ : @tensor.Tensor,
  beta~ : @tensor.Tensor,
  mean~ : @tensor.Tensor,
  var_~ : @tensor.Tensor,
  eps~ : Double = 1.0e-5
) -> BatchNorm2d {
  guard gamma.ndim() == 1 else {
    abort("BatchNorm2d: gamma must be 1D")
  }
  let c = gamma.shape()[0]
  guard beta.ndim() == 1 && beta.shape()[0] == c else {
    abort("BatchNorm2d: beta shape mismatch")
  }
  guard mean.ndim() == 1 && mean.shape()[0] == c else {
    abort("BatchNorm2d: mean shape mismatch")
  }
  guard var_.ndim() == 1 && var_.shape()[0] == c else {
    abort("BatchNorm2d: var shape mismatch")
  }
  { gamma, beta, running_mean: mean, running_var: var_, eps }
}

///|
/// Forward pass: applies batch normalization per channel (inference mode).
/// Input: [N, C, H, W], Output: [N, C, H, W]
pub fn BatchNorm2d::forward(
  self : BatchNorm2d,
  input : @autograd.Variable
) -> @autograd.Variable {
  let data = input.data()
  guard data.ndim() == 4 else {
    abort("BatchNorm2d::forward: input must be 4D (NCHW)")
  }
  let n = data.shape()[0]
  let c = data.shape()[1]
  let h = data.shape()[2]
  let w = data.shape()[3]
  let total = n * c * h * w
  let result = Array::make(total, 0.0)
  for ch = 0; ch < c; ch = ch + 1 {
    let mean = self.running_mean.get([ch])
    let var_ = self.running_var.get([ch])
    let g = self.gamma.get([ch])
    let b = self.beta.get([ch])
    let inv_std = 1.0 / (var_ + self.eps).sqrt()
    for batch = 0; batch < n; batch = batch + 1 {
      for row = 0; row < h; row = row + 1 {
        for col = 0; col < w; col = col + 1 {
          let idx = ((batch * c + ch) * h + row) * w + col
          let x = data.get([batch, ch, row, col])
          result[idx] = (x - mean) * inv_std * g + b
        }
      }
    }
  }
  let out = @tensor.Tensor::from_array_with_shape(result, [n, c, h, w])
  @autograd.Variable::new(out, requires_grad=false)
}

///|
/// BatchNorm2d has no trainable parameters in inference mode.
pub fn BatchNorm2d::parameters(self : BatchNorm2d) -> Array[@autograd.Variable] {
  ignore(self)
  []
}
