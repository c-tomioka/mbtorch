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

///|
/// Simple pseudo-random tensor with values in [-1, 1), using LCG.
pub fn Tensor::rand(shape : Array[Int], seed? : UInt64 = 42UL) -> Tensor {
  let mut total = 1
  for i = 0; i < shape.length(); i = i + 1 {
    total = total * shape[i]
  }
  let data = Array::make(total, 0.0)
  let mut state = seed
  for i = 0; i < total; i = i + 1 {
    // LCG: state = state * 6364136223846793005 + 1442695040888963407
    state = state * 6364136223846793005UL + 1442695040888963407UL
    // Extract bits and map to [-1, 1)
    let bits = (state >> 33).to_int()
    data[i] = bits.to_double() / 1073741824.0 - 1.0
  }
  { data_: data, shape_: shape.copy() }
}
