///|
fn main {
  // 1. Constructors
  println("=== Constructors ===")
  let s = @tensor.Tensor::scalar(42.0)
  println("scalar:        \{s}")
  let v = @tensor.Tensor::from_array([1.0, 2.0, 3.0])
  println("from_array:    \{v}")
  let m = @tensor.Tensor::from_array_2d([[1.0, 2.0], [3.0, 4.0]])
  println("from_array_2d: \{m}")
  let z = @tensor.Tensor::zeros([2, 3])
  println("zeros([2,3]):  \{z}")
  let o = @tensor.Tensor::ones([3])
  println("ones([3]):     \{o}")

  // 2. Element-wise add / mul (also works via + and * operators)
  println("\n=== Element-wise ops ===")
  let a = @tensor.Tensor::from_array([1.0, 2.0, 3.0])
  let b = @tensor.Tensor::from_array([4.0, 5.0, 6.0])
  println("a:     \{a}")
  println("b:     \{b}")
  println("a + b: \{a + b}")
  println("a * b: \{a * b}")

  // 3. Matrix multiplication
  println("\n=== Matmul ===")
  let x = @tensor.Tensor::from_array_2d([[1.0, 2.0], [3.0, 4.0]])
  let y = @tensor.Tensor::from_array_2d([[5.0, 6.0], [7.0, 8.0]])
  println("x:            \{x}")
  println("y:            \{y}")
  println("x.matmul(y):  \{x.matmul(y)}")
}
