# 0010: ONNX Export MVP (Sequential Models)

* Status: proposed
* Date: 2026-03-10
* Deciders: @c-tomioka, Claude Code

## Context and Problem Statement

MbTorch は ONNX フォーマットからの学習済みモデルのインポートに対応しているが（ADR-0007, ADR-0009）、MbTorch モデルを ONNX に書き出すエクスポート機能がない。エクスポートが必要な理由:
- **相互運用性**: MbTorch で学習したモデルを onnxruntime、TensorRT 等で実行できるようにする
- **検証**: ラウンドトリップ（エクスポート → インポート）テストにより正確性を検証する
- **デプロイ**: MbTorch での学習結果を最適化された推論ランタイムへ展開する

ADR-0007 ではエクスポートを後のフェーズに明示的に延期していた。本 ADR でその MVP スコープを定義する。

## Decision Drivers

- インポートとの対称性: インポートが扱うレイヤー型と同じものをサポートする
- 既存の ONNX 型（OnnxGraph, OnnxNode, OnnxTensor）を中間表現として再利用する
- 既存の protobuf エンコードヘルパーを再利用する（テストコードから本番コードへ昇格）
- onnxruntime でロード可能な ONNX ファイルを生成する（opset 13）
- MVP ではシーケンシャルモデルのみ対応（インポートの制約と同等）
- ラウンドトリップの正確性: エクスポート → parse_onnx → ロードで同一モデルが得られること

## Decision Outcome

シーケンシャルモデル（`Array[@nn.Layer]`）の ONNX エクスポートを、単一の公開関数 `export_onnx(layers, input_shape) -> Bytes` で実装する。protobuf エンコードヘルパーをテストコードから `io/onnx_encode.mbt` の本番コードへ昇格させる。

### 公開 API

- `export_onnx(layers: Array[@nn.Layer], input_shape: Array[Int]) -> Bytes raise OnnxError`

### レイヤーから ONNX への変換マッピング

| MbTorch レイヤー | ONNX Op | 備考 |
|---|---|---|
| Linear | Gemm (transB=1) | weight を [in, out] から [out, in] に転置して格納 |
| Conv2d | Conv | strides, pads, dilations=[1,1], group=1 属性 |
| BatchNorm2d | BatchNormalization | 5 入力 (X, gamma, beta, mean, var)、epsilon 属性 |
| Relu | Relu | 属性なし |
| Flatten | Flatten | axis=1 |

### MVP スコープ

含める:
- Layer enum の全 5 バリアント
- float32 のみ（ONNX dtype=1）
- 静的 shape
- シーケンシャルモデル（`Array[@nn.Layer]`）
- ONNX opset 13
- 重みはグラフ初期化子として埋め込み

含めない:
- 動的 shape / シンボリック次元
- 非シーケンシャル（DAG）グラフ
- float32 以外の dtype（float16, int8, bfloat16）
- MaxPool2d, AvgPool2d, Dropout, Attention, RNN
- groups != 1 の Conv
- ONNX 外部データフォーマット
- safetensors エクスポート
- ONNX メタデータ（producer_name, doc_string）

### アーキテクチャ

- `io/onnx_encode.mbt`: Protobuf エンコードプリミティブ（`io/onnx_test.mbt` から昇格）
- `io/export.mbt`: エクスポートロジック（`layers_to_onnx_graph` + `export_onnx`）
- `io/export_test.mbt`: 9 テスト（構造検証 + ラウンドトリップ）

データフロー: `Array[@nn.Layer]` → `OnnxGraph`（既存型を再利用）→ protobuf バイト列 → `Bytes`

### 後方互換性

- 既存のインポート API に変更なし
- 既存の ONNX 型に変更なし
- エンコードヘルパーをテストから本番コードへ移動（テストコードはリネームしたコピーを使用）

## Alternatives Considered

### A: ONNX エクスポートを実装しない

- 利点: 新規コードなし、メンテナンス不要
- 欠点: 外部ランタイムとの相互運用性なし、ラウンドトリップテスト不可

### B: より広い ONNX サブセットを最初からサポートする

- 利点: より広い互換性
- 欠点: 作業量が大幅に増加、MbTorch のターゲットユースケースに不要な ONNX オペレーターが多い

### C: 別フォーマット（例: safetensors のみ）にエクスポートする

- 利点: よりシンプルなバイナリフォーマット
- 欠点: グラフトポロジー情報が失われる、推論用途での広いサポートがない

## Consequences

### Pros

- ラウンドトリップテスト（エクスポート → インポート → 比較）が可能になる
- onnxruntime / TensorRT デプロイへの道が開ける
- 既存の型とエンコードコードを再利用
- API サーフェスが小さい（公開関数 1 個）

### Cons

- シーケンシャルモデルのみの制約（インポートと同等）
- ValueInfoProto の型/shape メタデータなし（onnxruntime がデータから推論する）

## Links

- ADR-0007: Phase 1 ONNX + safetensors（インポート）
- ADR-0009: Conv2d + BatchNorm2d MVP
- コード: `io/onnx_encode.mbt`, `io/export.mbt`, `io/export_test.mbt`
