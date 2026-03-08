///|
/// Deserialize a .mbt JSON string to a single Linear layer.
pub fn deserialize_linear(json_str : String) -> @nn.Linear raise MbtFormatError {
  let layers = deserialize_model(json_str)
  guard layers.length() == 1 else {
    raise MbtFormatError::InvalidFormat(
      "deserialize_linear: expected 1 layer, got \{layers.length()}",
    )
  }
  layers[0].1
}

///|
/// Deserialize a .mbt JSON string to an array of named Linear layers.
pub fn deserialize_model(
  json_str : String,
) -> Array[(String, @nn.Linear)] raise MbtFormatError {
  let json = @json.parse(json_str) catch {
    _ => raise MbtFormatError::InvalidFormat("failed to parse JSON")
  }
  let model : MbtModel = @json.from_json(json) catch {
    _ =>
      raise MbtFormatError::InvalidFormat(
        "JSON structure does not match .mbt format",
      )
  }
  // Validate format
  guard model.format == "mbtorch" else {
    raise MbtFormatError::InvalidFormat(
      "expected format \"mbtorch\", got \"\{model.format}\"",
    )
  }
  // Validate version
  guard model.version == 1 else {
    raise MbtFormatError::UnsupportedVersion(model.version)
  }
  // Reconstruct layers
  let result : Array[(String, @nn.Linear)] = []
  for i = 0; i < model.layers.length(); i = i + 1 {
    let ld = model.layers[i]
    guard ld.layer_type == "Linear" else {
      raise MbtFormatError::InvalidFormat(
        "unsupported layer type: \{ld.layer_type}",
      )
    }
    let weight = @tensor.Tensor::from_array_with_shape(
      ld.weight.data,
      ld.weight.shape,
    )
    let bias = @tensor.Tensor::from_array_with_shape(
      ld.bias.data,
      ld.bias.shape,
    )
    let linear = @nn.Linear::from_tensors(weight, bias)
    result.push((ld.name, linear))
  }
  result
}
