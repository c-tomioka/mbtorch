///|
/// Heterogeneous layer type for building models with mixed layer types.
pub(all) enum Layer {
  Linear(Linear)
  Conv2d(Conv2d)
  BatchNorm2d(BatchNorm2d)
  Relu
  Flatten
  SelfAttention(SelfAttention)
}

///|
/// Forward pass dispatched by layer type.
pub fn Layer::forward(self : Layer, input : @autograd.Variable) -> @autograd.Variable {
  match self {
    Linear(l) => l.forward(input)
    Conv2d(c) => c.forward(input)
    BatchNorm2d(bn) => bn.forward(input)
    Relu => input.relu()
    Flatten => {
      let data = input.data()
      let n = data.shape()[0]
      let flat_size = data.size() / n
      let reshaped = data.reshape([n, flat_size])
      @autograd.Variable::new(reshaped, requires_grad=false)
    }
    SelfAttention(sa) => sa.forward(input)
  }
}

///|
/// Collect trainable parameters from the layer.
pub fn Layer::parameters(self : Layer) -> Array[@autograd.Variable] {
  match self {
    Linear(l) => l.parameters()
    Conv2d(c) => c.parameters()
    BatchNorm2d(bn) => bn.parameters()
    Relu => []
    Flatten => []
    SelfAttention(sa) => sa.parameters()
  }
}
