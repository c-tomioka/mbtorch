///|
/// Numerically check gradients by comparing autograd with finite differences.
/// Returns true if all relative errors are below tolerance.
pub fn grad_check(
  f : (Array[Variable]) -> Variable,
  inputs : Array[Variable],
  epsilon? : Double = 1.0e-5,
  tolerance? : Double = 1.0e-4,
) -> Bool {
  // 1. Forward + backward to get autograd gradients
  let output = f(inputs)
  output.backward()
  // 2. For each input, compare autograd grad with numerical grad
  for idx = 0; idx < inputs.length(); idx = idx + 1 {
    let input = inputs[idx]
    guard input.requires_grad else { continue idx + 1 }
    let analytic_grad = match input.grad {
      Some(g) => g
      None => return false
    }
    let n = input.data.size()
    for i = 0; i < n; i = i + 1 {
      // Perturb element i by +epsilon and -epsilon
      let val = get_flat(input.data, i)
      // f(x + eps)
      let inputs_plus = make_perturbed_inputs(inputs, idx, i, val + epsilon)
      let out_plus = f(inputs_plus)
      let f_plus = out_plus.data.get([])
      // f(x - eps)
      let inputs_minus = make_perturbed_inputs(inputs, idx, i, val - epsilon)
      let out_minus = f(inputs_minus)
      let f_minus = out_minus.data.get([])
      // Numerical gradient
      let numerical = (f_plus - f_minus) / (2.0 * epsilon)
      let analytic = get_flat(analytic_grad, i)
      // Relative error
      let diff = abs_val(analytic - numerical)
      let scale = max_val(abs_val(analytic), abs_val(numerical)) + 1.0e-8
      if diff / scale > tolerance {
        return false
      }
    }
  }
  true
}

///|
fn get_flat(t : @tensor.Tensor, i : Int) -> Double {
  // Access flat element i regardless of shape
  // For scalar: i must be 0
  if t.ndim() == 0 {
    return t.get([])
  }
  // For 1D:
  if t.ndim() == 1 {
    return t.get([i])
  }
  // For 2D: convert flat index to row, col
  let cols = t.shape()[1]
  let row = i / cols
  let col = i % cols
  t.get([row, col])
}

///|
fn make_perturbed_inputs(
  inputs : Array[Variable],
  input_idx : Int,
  elem_idx : Int,
  new_val : Double,
) -> Array[Variable] {
  let result : Array[Variable] = []
  for idx = 0; idx < inputs.length(); idx = idx + 1 {
    if idx == input_idx {
      let perturbed = perturb_tensor(inputs[idx].data, elem_idx, new_val)
      result.push(Variable::new(perturbed))
    } else {
      result.push(Variable::new(inputs[idx].data))
    }
  }
  result
}

///|
fn perturb_tensor(
  t : @tensor.Tensor,
  flat_idx : Int,
  new_val : Double,
) -> @tensor.Tensor {
  let shape = t.shape()
  let n = t.size()
  let data = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    data[i] = get_flat(t, i)
  }
  data[flat_idx] = new_val
  // Reconstruct tensor with correct shape
  if shape.length() == 0 {
    return @tensor.Tensor::scalar(data[0])
  }
  if shape.length() == 1 {
    return @tensor.Tensor::from_array(data)
  }
  // 2D: reshape flat data back
  let rows = shape[0]
  let cols = shape[1]
  let rows_arr : Array[Array[Double]] = []
  for r = 0; r < rows; r = r + 1 {
    let row : Array[Double] = []
    for c = 0; c < cols; c = c + 1 {
      row.push(data[r * cols + c])
    }
    rows_arr.push(row)
  }
  @tensor.Tensor::from_array_2d(rows_arr)
}

///|
fn abs_val(x : Double) -> Double {
  if x < 0.0 {
    -x
  } else {
    x
  }
}

///|
fn max_val(a : Double, b : Double) -> Double {
  if a > b {
    a
  } else {
    b
  }
}
