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
}
