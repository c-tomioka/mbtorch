# 0009: Conv2d + BatchNorm2d MVP (ONNX CNN インポート対応)

* Status: proposed
* Date: 2026-03-09
* Deciders: @c-tomioka, Claude Code

## Context and Problem Statement

MbTorch の ONNX インポートは MLP (Gemm/MatMul/Add/Relu) のみ対応しており、小さな CNN モデル（手書き数字認識等）をインポートできない。ブラウザ/エッジ環境で CNN 推論を実行するため、Conv2d と BatchNormalization を最小範囲で追加する。

## Decision Drivers

- PyTorch で学習した小規模 CNN を MbTorch にインポートしたい
- 推論のみ（backward なし）で MVP とする
- 既存の MLP インポート API を変更しない
- 既存のレイヤリング規則（core → nn → io）を維持する

## Decision Outcome

推論のみの Conv2d + BatchNorm2d を追加し、Layer enum で異種レイヤーを統一的に扱う。

### 新規 API

#### core/tensor

- `Tensor::reshape(shape: Array[Int]) -> Tensor` — 要素数一致検証のみ
- `Tensor::conv2d(kernel, bias?, stride, pad) -> Tensor` — 直接ループ実装

#### nn

- `Conv2d` struct — `from_tensors(weight, bias?, stride, padding)` + `forward` + `parameters`
- `BatchNorm2d` struct — `from_tensors(gamma, beta, mean, var_, eps)` + `forward` + `parameters`
- `Layer` enum — `Linear | Conv2d | BatchNorm2d | Relu | Flatten` + `forward` + `parameters`

#### io

- `OnnxAttribute` に `ints: Array[Int]` フィールド追加
- `load_cnn_model_from_onnx_and_safetensors(onnx, safetensors) -> Array[(String, Layer)]`

### MVP スコープ

含める:
- Conv2d: groups=1, dilations=[1,1], 任意の stride/padding
- BatchNorm2d: 推論のみ（running_mean/var 凍結）
- Flatten: reshape による N次元 → 2次元変換
- Layer enum による異種レイヤーモデル構築
- ONNX Conv/BatchNormalization/Relu/Flatten/Gemm/MatMul ノード対応

含めない:
- backward / training
- MaxPool2d, AvgPool2d
- im2col 最適化
- grouped / depthwise conv
- auto_pad (SAME_UPPER, SAME_LOWER)
- BN folding（Conv への融合）

### 後方互換性

- 既存の `load_model_from_onnx_and_safetensors` は変更なし
- 新関数 `load_cnn_model_from_onnx_and_safetensors` を追加

## Pros and Cons

### Pros

- PyTorch CNN モデルの推論が MbTorch で可能になる
- Layer enum により将来のレイヤー追加が容易
- 既存 API に影響なし

### Cons

- backward 未対応のため CNN の学習は不可
- 直接ループ実装のため大きなモデルでは遅い（将来 im2col で最適化）
- groups/dilations の制約あり

## Links

- ADR-0007: Phase 1 ONNX + safetensors
- Code: `core/tensor/ops_conv.mbt`, `nn/conv2d.mbt`, `nn/batchnorm2d.mbt`, `nn/layer.mbt`
- Code: `io/import.mbt` (load_cnn_model_from_onnx_and_safetensors)
- Example: `examples/import_cnn/`
