///|
/// Serialize a single Linear layer to a .mbt JSON string.
pub fn serialize_linear(
  layer : @nn.Linear,
  name? : String = "linear",
) -> String {
  let model = make_mbt_model([(name, layer)])
  model.to_json().stringify()
}

///|
/// Serialize multiple named Linear layers to a .mbt JSON string.
pub fn serialize_model(layers : Array[(String, @nn.Linear)]) -> String {
  let model = make_mbt_model(layers)
  model.to_json().stringify()
}

///|
fn make_mbt_model(layers : Array[(String, @nn.Linear)]) -> MbtModel {
  let layer_data : Array[LayerData] = []
  for i = 0; i < layers.length(); i = i + 1 {
    let (name, layer) = layers[i]
    let w = layer.weight.data()
    let b = layer.bias.data()
    let ws = w.shape()
    let bs = b.shape()
    layer_data.push({
      name,
      layer_type: "Linear",
      in_features: ws[0],
      out_features: ws[1],
      weight: { shape: ws, data: w.to_array() },
      bias: { shape: bs, data: b.to_array() },
    })
  }
  { format: "mbtorch", version: 1, layers: layer_data }
}
