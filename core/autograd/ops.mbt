///|
pub fn Variable::add(self : Variable, other : Variable) -> Variable {
  let result_data = self.data.add(other.data)
  let requires_grad = self.requires_grad || other.requires_grad
  let grad_fn = if requires_grad {
    Some(AddBackward(self, other))
  } else {
    None
  }
  { data: result_data, grad: None, grad_fn, requires_grad }
}

///|
pub fn Variable::mul(self : Variable, other : Variable) -> Variable {
  let result_data = self.data.mul(other.data)
  let requires_grad = self.requires_grad || other.requires_grad
  let grad_fn = if requires_grad {
    Some(MulBackward(self, other))
  } else {
    None
  }
  { data: result_data, grad: None, grad_fn, requires_grad }
}

///|
pub fn Variable::matmul(self : Variable, other : Variable) -> Variable {
  let result_data = self.data.matmul(other.data)
  let requires_grad = self.requires_grad || other.requires_grad
  let grad_fn = if requires_grad {
    Some(MatmulBackward(self, other))
  } else {
    None
  }
  { data: result_data, grad: None, grad_fn, requires_grad }
}

///|
pub fn Variable::sub(self : Variable, other : Variable) -> Variable {
  let result_data = self.data.sub(other.data)
  let requires_grad = self.requires_grad || other.requires_grad
  let grad_fn = if requires_grad {
    Some(SubBackward(self, other))
  } else {
    None
  }
  { data: result_data, grad: None, grad_fn, requires_grad }
}

///|
pub fn Variable::sum(self : Variable) -> Variable {
  let result_data = self.data.sum()
  let grad_fn = if self.requires_grad { Some(SumBackward(self)) } else { None }
  { data: result_data, grad: None, grad_fn, requires_grad: self.requires_grad }
}

///|
/// Expand a 1D Variable [cols] to 2D [rows, cols] by repeating.
/// Backward sums gradients along the batch (row) dimension.
pub fn Variable::expand_rows(self : Variable, rows : Int) -> Variable {
  let cols = self.shape()[0]
  let rows_arr : Array[Array[Double]] = []
  for _r = 0; _r < rows; _r = _r + 1 {
    let row : Array[Double] = []
    for c = 0; c < cols; c = c + 1 {
      row.push(self.data.get([c]))
    }
    rows_arr.push(row)
  }
  let tensor = @tensor.Tensor::from_array_2d(rows_arr)
  let grad_fn = if self.requires_grad {
    Some(ExpandRowsBackward(self))
  } else {
    None
  }
  { data: tensor, grad: None, grad_fn, requires_grad: self.requires_grad }
}

///|
pub fn Variable::relu(self : Variable) -> Variable {
  let n = self.data.size()
  let shape = self.data.shape()
  let result_data = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    let val = get_flat_val(self.data, i)
    result_data[i] = if val > 0.0 { val } else { 0.0 }
  }
  let tensor = rebuild_tensor(result_data, shape)
  let grad_fn = if self.requires_grad { Some(ReluBackward(self)) } else { None }
  { data: tensor, grad: None, grad_fn, requires_grad: self.requires_grad }
}

///|
pub fn Variable::tanh(self : Variable) -> Variable {
  let n = self.data.size()
  let shape = self.data.shape()
  let result_data = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    let val = get_flat_val(self.data, i)
    result_data[i] = tanh_scalar(val)
  }
  let tensor = rebuild_tensor(result_data, shape)
  let grad_fn = if self.requires_grad { Some(TanhBackward(self)) } else { None }
  { data: tensor, grad: None, grad_fn, requires_grad: self.requires_grad }
}

///|
pub fn Variable::sigmoid(self : Variable) -> Variable {
  let n = self.data.size()
  let shape = self.data.shape()
  let result_data = Array::make(n, 0.0)
  for i = 0; i < n; i = i + 1 {
    let val = get_flat_val(self.data, i)
    result_data[i] = sigmoid_scalar(val)
  }
  let tensor = rebuild_tensor(result_data, shape)
  let grad_fn = if self.requires_grad {
    Some(SigmoidBackward(self))
  } else {
    None
  }
  { data: tensor, grad: None, grad_fn, requires_grad: self.requires_grad }
}

///|
fn tanh_scalar(x : Double) -> Double {
  let e2x = @math.exp(2.0 * x)
  (e2x - 1.0) / (e2x + 1.0)
}

///|
fn sigmoid_scalar(x : Double) -> Double {
  1.0 / (1.0 + @math.exp(-x))
}

///|
fn get_flat_val(t : @tensor.Tensor, i : Int) -> Double {
  match t.ndim() {
    0 => t.get([])
    1 => t.get([i])
    _ => {
      let cols = t.shape()[1]
      t.get([i / cols, i % cols])
    }
  }
}

///|
fn rebuild_tensor(data : Array[Double], shape : Array[Int]) -> @tensor.Tensor {
  match shape.length() {
    0 => @tensor.Tensor::scalar(data[0])
    1 => @tensor.Tensor::from_array(data)
    _ => {
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
  }
}
