///|
/// Error type for .mbt format issues
pub(all) suberror MbtFormatError {
  InvalidFormat(String)
  UnsupportedVersion(Int)
  MissingField(String)
  ShapeMismatch(String, Array[Int], Array[Int])
}

///|
/// Error type for safetensors format issues
pub(all) suberror SafetensorsError {
  InvalidHeader(String)
  UnsupportedDtype(String)
  DataSizeMismatch(String)
}

///|
/// Metadata for a single tensor in a safetensors file
pub(all) struct SafetensorsMeta {
  name : String
  dtype : String
  shape : Array[Int]
  data_start : Int
  data_end : Int
}

///|
/// DTO for tensor data in JSON
priv struct TensorData {
  shape : Array[Int]
  data : Array[Double]
} derive(ToJson, FromJson)

///|
/// DTO for a single layer's data in JSON
priv struct LayerData {
  name : String
  layer_type : String
  in_features : Int
  out_features : Int
  weight : TensorData
  bias : TensorData
} derive(ToJson, FromJson)

///|
/// Top-level DTO for the .mbt model format
priv struct MbtModel {
  format : String
  version : Int
  layers : Array[LayerData]
} derive(ToJson, FromJson)
