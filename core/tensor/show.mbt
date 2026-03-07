///|
pub impl Show for Tensor with output(self, logger) {
  logger.write_string("Tensor(shape=\{self.shape_}, data=\{self.data_})")
}
