///|
/// Parse a JSON string containing a 2D array of numbers.
fn parse_2d_array(json_str : String) -> Array[Array[Double]] {
  let json = @json.parse(json_str) catch { _ => abort("JSON parse failed") }
  match json {
    Array(rows) => {
      let result : Array[Array[Double]] = []
      for i = 0; i < rows.length(); i = i + 1 {
        match rows[i] {
          Array(cols) => {
            let row : Array[Double] = []
            for j = 0; j < cols.length(); j = j + 1 {
              match cols[j] {
                Number(n, ..) => row.push(n)
                _ => abort("expected number in expected_output.json")
              }
            }
            result.push(row)
          }
          _ => abort("expected array of arrays in expected_output.json")
        }
      }
      result
    }
    _ => abort("expected 2D array in expected_output.json")
  }
}

///|
/// Import a PyTorch-exported CNN (ONNX + safetensors) and verify
/// that MbTorch forward output matches the PyTorch expected output.
fn main {
  // 1. Read model files
  let onnx_bytes = @fs.read_file_to_bytes(
    "examples/import_cnn/model.onnx",
  ) catch {
    _ => abort("failed to read model.onnx — run export_from_pytorch.py first")
  }
  let st_bytes = @fs.read_file_to_bytes(
    "examples/import_cnn/model.safetensors",
  ) catch {
    _ =>
      abort(
        "failed to read model.safetensors — run export_from_pytorch.py first",
      )
  }
  // 2. Build MbTorch model from ONNX structure + safetensors weights
  let layers = @io.load_cnn_model_from_onnx_and_safetensors(
    onnx_bytes, st_bytes,
  ) catch {
    _ => abort("failed to load CNN model from ONNX + safetensors")
  }
  println("Loaded \{layers.length()} layers from ONNX + safetensors")
  for i = 0; i < layers.length(); i = i + 1 {
    let (name, _layer) = layers[i]
    println("  [\{i}] \{name}")
  }
  // 3. Forward pass: input N=2, C=1, H=4, W=4 (values 1..32)
  let input_data : Array[Double] = []
  for i = 1; i <= 32; i = i + 1 {
    input_data.push(i.to_double())
  }
  let mut x = @autograd.Variable::new(
    @tensor.Tensor::from_array_with_shape(input_data, [2, 1, 4, 4]),
  )
  // Run through all layers sequentially
  for i = 0; i < layers.length(); i = i + 1 {
    x = layers[i].1.forward(x)
  }
  println("Output shape: \{x.shape()}")
  // 4. Read expected output
  let expected_json = @fs.read_file_to_string(
    "examples/import_cnn/expected_output.json",
  ) catch {
    _ =>
      abort(
        "failed to read expected_output.json — run export_from_pytorch.py first",
      )
  }
  let expected = parse_2d_array(expected_json)
  // 5. Compare
  let out_rows = x.data().shape()[0]
  let out_cols = x.data().shape()[1]
  if expected.length() != out_rows {
    println(
      "NG: shape mismatch — expected \{expected.length()} rows, got \{out_rows}",
    )
    return
  }
  if expected[0].length() != out_cols {
    println(
      "NG: shape mismatch — expected \{expected[0].length()} cols, got \{out_cols}",
    )
    return
  }
  let eps = 1.0e-4
  let mut max_diff = 0.0
  for i = 0; i < out_rows; i = i + 1 {
    for j = 0; j < out_cols; j = j + 1 {
      let mbtorch_val = x.data().get([i, j])
      let pytorch_val = expected[i][j]
      let diff = (mbtorch_val - pytorch_val).abs()
      if diff > max_diff {
        max_diff = diff
      }
      if diff > eps {
        println(
          "NG: value mismatch at [\{i}, \{j}] — MbTorch=\{mbtorch_val}, PyTorch=\{pytorch_val}, diff=\{diff}",
        )
        return
      }
    }
  }
  println("Max absolute difference: \{max_diff}")
  println("OK")
}
