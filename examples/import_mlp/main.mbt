///|
/// Parse a JSON string containing a 2D array of numbers.
/// Expected format: [[1.0, 2.0], [3.0, 4.0]]
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
/// Import a PyTorch-exported MLP (ONNX + safetensors) and verify
/// that MbTorch forward output matches the PyTorch expected output.
fn main {
  // 1. Read model files
  let onnx_bytes = @fs.read_file_to_bytes("examples/import_mlp/model.onnx") catch {
    _ => abort("failed to read model.onnx — run export_from_pytorch.py first")
  }
  let st_bytes = @fs.read_file_to_bytes("examples/import_mlp/model.safetensors") catch {
    _ =>
      abort(
        "failed to read model.safetensors — run export_from_pytorch.py first",
      )
  }
  // 2. Build MbTorch model from ONNX structure + safetensors weights
  let layers = @io.load_model_from_onnx_and_safetensors(onnx_bytes, st_bytes) catch {
    _ => abort("failed to load model from ONNX + safetensors")
  }
  println("Loaded \{layers.length()} layers from ONNX + safetensors")
  // 3. Forward pass (same input as export_from_pytorch.py)
  let x = @autograd.Variable::new(
    @tensor.Tensor::from_array_2d([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]),
  )
  // Model structure: Linear -> ReLU -> Linear
  let h = layers[0].1.forward(x).relu()
  let y = layers[1].1.forward(h)
  // 4. Read expected output
  let expected_json = @fs.read_file_to_string(
    "examples/import_mlp/expected_output.json",
  ) catch {
    _ =>
      abort(
        "failed to read expected_output.json — run export_from_pytorch.py first",
      )
  }
  let expected = parse_2d_array(expected_json)
  // 5. Compare shapes
  let out_rows = y.data().shape()[0]
  let out_cols = y.data().shape()[1]
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
  // 6. Compare values element-wise
  let eps = 1.0e-5
  let mut max_diff = 0.0
  for i = 0; i < out_rows; i = i + 1 {
    for j = 0; j < out_cols; j = j + 1 {
      let mbtorch_val = y.data().get([i, j])
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
