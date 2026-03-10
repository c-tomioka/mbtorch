///|
struct Variable {
  mut data : @tensor.Tensor
  mut grad : @tensor.Tensor?
  grad_fn : GradFn?
  requires_grad : Bool
}

///|
priv enum GradFn {
  AddBackward(Variable, Variable)
  MulBackward(Variable, Variable)
  MatmulBackward(Variable, Variable)
  SumBackward(Variable)
  SubBackward(Variable, Variable)
  ReluBackward(Variable)
  TanhBackward(Variable)
  SigmoidBackward(Variable)
  ExpandRowsBackward(Variable)
  ScaleBackward(Variable, Double)
  SoftmaxBackward(Variable, @tensor.Tensor) // (input, softmax_output_data)
  BatchedMatmulBackward(Variable, Variable)
  TransposeLast2Backward(Variable)
  ReshapeBackward(Variable, Array[Int]) // (input, original_shape)
}
