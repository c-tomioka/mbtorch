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
/// Error type for ONNX format issues
pub(all) suberror OnnxError {
  InvalidModel(String)
  UnsupportedOp(String)
  UnsupportedDtype(String)
  ShapeMismatch(String)
}

///|
/// Parsed ONNX tensor (initializer / weight)
pub(all) struct OnnxTensor {
  name : String
  dtype : Int // ONNX data_type enum (1=FLOAT)
  shape : Array[Int]
  data : Array[Double]
}

///|
/// Parsed ONNX attribute (e.g. transB=1)
pub(all) struct OnnxAttribute {
  name : String
  int_value : Int64
  float_value : Double
  ints : Array[Int]
}

///|
/// Parsed ONNX node (single op)
pub(all) struct OnnxNode {
  op_type : String
  inputs : Array[String]
  outputs : Array[String]
  attributes : Array[OnnxAttribute]
}

///|
/// Parsed ONNX graph
pub(all) struct OnnxGraph {
  nodes : Array[OnnxNode]
  initializers : Array[OnnxTensor]
  input_names : Array[String]
  output_names : Array[String]
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
