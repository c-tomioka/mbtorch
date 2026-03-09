///|
/// Reshape a tensor to a new shape. The total number of elements must match.
pub fn Tensor::reshape(self : Tensor, shape : Array[Int]) -> Tensor {
  let mut new_size = 1
  for i = 0; i < shape.length(); i = i + 1 {
    new_size = new_size * shape[i]
  }
  guard new_size == self.size() else {
    abort(
      "reshape: size mismatch — \{self.size()} elements cannot reshape to \{new_size}",
    )
  }
  Tensor::from_array_with_shape(self.to_array(), shape)
}

///|
/// 2D convolution on a 4D NCHW tensor.
/// Input: [N, C_in, H, W], kernel: [C_out, C_in, kH, kW]
/// Returns: [N, C_out, out_H, out_W]
/// pad: [pad_top, pad_left, pad_bottom, pad_right]
pub fn Tensor::conv2d(
  self : Tensor,
  kernel : Tensor,
  bias~ : Tensor? = None,
  stride~ : Array[Int] = [1, 1],
  pad~ : Array[Int] = [0, 0, 0, 0]
) -> Tensor {
  guard self.ndim() == 4 else { abort("conv2d: input must be 4D (NCHW)") }
  guard kernel.ndim() == 4 else {
    abort("conv2d: kernel must be 4D (C_out, C_in, kH, kW)")
  }
  let n = self.shape()[0]
  let c_in = self.shape()[1]
  let h = self.shape()[2]
  let w = self.shape()[3]
  let c_out = kernel.shape()[0]
  let k_c_in = kernel.shape()[1]
  let kh = kernel.shape()[2]
  let kw = kernel.shape()[3]
  guard c_in == k_c_in else {
    abort(
      "conv2d: input channels \{c_in} != kernel channels \{k_c_in}",
    )
  }
  let stride_h = stride[0]
  let stride_w = stride[1]
  let pad_top = pad[0]
  let pad_left = pad[1]
  let pad_bottom = pad[2]
  let pad_right = pad[3]
  let h_padded = h + pad_top + pad_bottom
  let w_padded = w + pad_left + pad_right
  let out_h = (h_padded - kh) / stride_h + 1
  let out_w = (w_padded - kw) / stride_w + 1
  guard out_h > 0 && out_w > 0 else {
    abort("conv2d: output spatial dims must be positive")
  }
  let out_size = n * c_out * out_h * out_w
  let result = Array::make(out_size, 0.0)
  // Direct convolution loops
  for batch = 0; batch < n; batch = batch + 1 {
    for oc = 0; oc < c_out; oc = oc + 1 {
      for oh = 0; oh < out_h; oh = oh + 1 {
        for ow = 0; ow < out_w; ow = ow + 1 {
          let mut sum = 0.0
          for ic = 0; ic < c_in; ic = ic + 1 {
            for fh = 0; fh < kh; fh = fh + 1 {
              for fw = 0; fw < kw; fw = fw + 1 {
                let ih = oh * stride_h + fh - pad_top
                let iw = ow * stride_w + fw - pad_left
                if ih >= 0 && ih < h && iw >= 0 && iw < w {
                  let in_idx = ((batch * c_in + ic) * h + ih) * w + iw
                  let k_idx = ((oc * k_c_in + ic) * kh + fh) * kw + fw
                  sum = sum + self.data_[in_idx] * kernel.data_[k_idx]
                }
              }
            }
          }
          // Add bias if present
          match bias {
            Some(b) => sum = sum + b.data_[oc]
            None => ()
          }
          let out_idx = ((batch * c_out + oc) * out_h + oh) * out_w + ow
          result[out_idx] = sum
        }
      }
    }
  }
  Tensor::from_array_with_shape(result, [n, c_out, out_h, out_w])
}
