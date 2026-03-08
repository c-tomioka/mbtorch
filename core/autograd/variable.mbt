///|
pub fn Variable::new(
  data : @tensor.Tensor,
  requires_grad? : Bool = false,
) -> Variable {
  { data, grad: None, grad_fn: None, requires_grad }
}

///|
pub fn Variable::data(self : Variable) -> @tensor.Tensor {
  self.data
}

///|
pub fn Variable::grad(self : Variable) -> @tensor.Tensor? {
  self.grad
}

///|
pub fn Variable::shape(self : Variable) -> Array[Int] {
  self.data.shape()
}

///|
pub fn Variable::zero_grad(self : Variable) -> Unit {
  self.grad = None
}

///|
pub fn Variable::set_data(self : Variable, data : @tensor.Tensor) -> Unit {
  self.data = data
}
