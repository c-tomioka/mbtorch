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
pub fn Variable::sum(self : Variable) -> Variable {
  let result_data = self.data.sum()
  let grad_fn = if self.requires_grad { Some(SumBackward(self)) } else { None }
  { data: result_data, grad: None, grad_fn, requires_grad: self.requires_grad }
}
