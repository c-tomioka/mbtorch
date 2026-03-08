///|
struct Variable {
  data : @tensor.Tensor
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
}
