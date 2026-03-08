///|
pub fn Tensor::scalar(value : Double) -> Tensor {
  { data_: [value], shape_: [] }
}

///|
pub fn Tensor::from_array(data : Array[Double]) -> Tensor {
  { data_: data.copy(), shape_: [data.length()] }
}

///|
pub fn Tensor::from_array_2d(data : Array[Array[Double]]) -> Tensor {
  let rows = data.length()
  guard rows > 0 else { abort("from_array_2d: empty outer array") }
  let cols = data[0].length()
  let flat = Array::make(rows * cols, 0.0)
  for r = 0; r < rows; r = r + 1 {
    guard data[r].length() == cols else {
      abort("from_array_2d: inconsistent row lengths")
    }
    for c = 0; c < cols; c = c + 1 {
      flat[r * cols + c] = data[r][c]
    }
  }
  { data_: flat, shape_: [rows, cols] }
}

///|
pub fn Tensor::zeros(shape : Array[Int]) -> Tensor {
  let mut total = 1
  for i = 0; i < shape.length(); i = i + 1 {
    total = total * shape[i]
  }
  { data_: Array::make(total, 0.0), shape_: shape.copy() }
}

///|
pub fn Tensor::ones(shape : Array[Int]) -> Tensor {
  let mut total = 1
  for i = 0; i < shape.length(); i = i + 1 {
    total = total * shape[i]
  }
  { data_: Array::make(total, 1.0), shape_: shape.copy() }
}

///|
pub fn Tensor::fill(shape : Array[Int], value : Double) -> Tensor {
  let mut total = 1
  for i = 0; i < shape.length(); i = i + 1 {
    total = total * shape[i]
  }
  { data_: Array::make(total, value), shape_: shape.copy() }
}
