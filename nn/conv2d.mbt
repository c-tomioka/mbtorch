///|
/// Conv2d layer for inference only (no backward).
/// weight shape: [C_out, C_in, kH, kW]
/// bias shape:   [C_out] (optional)
pub struct Conv2d {
  weight : @autograd.Variable
  bias : @autograd.Variable?
  stride : Array[Int]
  padding : Array[Int]
}

///|
/// Create a Conv2d from pre-existing tensors.
/// weight: [C_out, C_in, kH, kW], bias: [C_out] (optional)
/// stride: [stride_h, stride_w]
/// padding: [pad_top, pad_left, pad_bottom, pad_right]
pub fn Conv2d::from_tensors(
  weight_data : @tensor.Tensor,
  bias_data~ : @tensor.Tensor? = None,
  stride~ : Array[Int] = [1, 1],
  padding~ : Array[Int] = [0, 0, 0, 0]
) -> Conv2d {
  guard weight_data.ndim() == 4 else {
    abort("Conv2d::from_tensors: weight must be 4D, got \{weight_data.ndim()}D")
  }
  let weight = @autograd.Variable::new(weight_data, requires_grad=false)
  let bias = match bias_data {
    Some(b) => {
      guard b.ndim() == 1 else {
        abort("Conv2d::from_tensors: bias must be 1D, got \{b.ndim()}D")
      }
      guard b.shape()[0] == weight_data.shape()[0] else {
        abort(
          "Conv2d::from_tensors: bias size \{b.shape()[0]} != C_out \{weight_data.shape()[0]}",
        )
      }
      Some(@autograd.Variable::new(b, requires_grad=false))
    }
    None => None
  }
  { weight, bias, stride, padding }
}

///|
/// Forward pass: applies 2D convolution (inference only, no grad_fn).
pub fn Conv2d::forward(self : Conv2d, input : @autograd.Variable) -> @autograd.Variable {
  let bias_tensor : @tensor.Tensor? = match self.bias {
    Some(b) => Some(b.data())
    None => None
  }
  let result = input.data().conv2d(
    self.weight.data(),
    bias=bias_tensor,
    stride=self.stride,
    pad=self.padding,
  )
  @autograd.Variable::new(result, requires_grad=false)
}

///|
pub fn Conv2d::parameters(self : Conv2d) -> Array[@autograd.Variable] {
  match self.bias {
    Some(b) => [self.weight, b]
    None => [self.weight]
  }
}
