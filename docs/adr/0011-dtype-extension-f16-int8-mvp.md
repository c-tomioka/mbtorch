# 0011: DType 拡張 — float16 / int8 MVP

* Status: proposed
* Date: 2026-03-10
* Deciders: @c-tomioka, Claude Code

## Context and Problem Statement

MbTorch のテンソルは内部的に `Array[Double]`（f64）でデータを保持し、IO 時のみ float32 へ変換している。Edge / ブラウザ環境ではモデルサイズと転送量がパフォーマンスのボトルネックになるため、より小さいデータ型のサポートが必要。

- float16: モデルサイズ約 1/2（float32 比）。GPU での混合精度推論にも利用
- int8: モデルサイズ約 1/4。Edge AI での省電力・高速推論の標準手法

MoonBit にはネイティブの Float16 / Int8 型がないため、UInt16 / Byte のビット操作による変換が必要。

## Decision Drivers

- Edge / ブラウザ向けのモデルサイズ削減
- 既存の演算パス（Double）への影響最小化
- ONNX / safetensors の float16 モデルとの相互運用性
- 既存バイナリ .mbt フォーマットとの後方互換性
- TDD による正確性の保証（変換精度のしきい値テスト）

## Decision Outcome

**ストレージ中心の dtype MVP** を実装する。

1. `DType` 列挙型 (`Float64 | Float32 | Float16 | Int8`) を `core/tensor` に追加
2. `Tensor` 構造体に `dtype_: DType` と `quant_params_: QuantParams?` フィールドを追加
3. **演算は全て Double のまま** — dtype は IO エンコーディングのみを制御
4. float16: IO 時に IEEE 754 半精度 ↔ Double の変換
5. int8: weight-only 線形量子化（scale + zero_point）。推論時に dequantize してから計算

### 公開 API

```moonbit
// core/tensor/dtype.mbt
pub(all) enum DType { Float64; Float32; Float16; Int8 }
pub(all) struct QuantParams { scale : Double; zero_point : Int }

// core/tensor/convert.mbt
pub fn double_to_float16_bits(value : Double) -> Int
pub fn float16_bits_to_double(bits : Int) -> Double
pub fn quantize_to_int8(data : Array[Double]) -> (Array[Int], Double, Int)
pub fn dequantize_from_int8(data : Array[Int], scale : Double, zero_point : Int) -> Array[Double]
pub fn Tensor::to_dtype(self : Tensor, target : DType) -> Tensor

// io/export.mbt
pub fn export_onnx_with_dtype(layers : Array[@nn.Layer], input_shape : Array[Int], dtype : String) -> Bytes

// io/binary.mbt
pub fn serialize_model_to_binary_with_dtype(layers : Array[(String, @tensor.Tensor)], dtype : String) -> Bytes
```

### MVP スコープ

含める:
- DType enum と QuantParams struct
- float16 ↔ Double 変換関数
- int8 ↔ Double 量子化/逆量子化関数
- `Tensor::to_dtype()` 公開 API
- Binary .mbt での f16/int8 シリアライズ/デシリアライズ
- ONNX import の float16 (dtype=10) 対応
- safetensors import の "F16" 対応
- ONNX export の float16 対応

含めない:
- 演算カーネルの float16/int8 ネイティブ対応
- activation 量子化（weight-only に限定）
- bfloat16, float8, int4
- ONNX int8 import/export（DequantizeLinear ノード対応が複雑）
- safetensors int8（量子化パラメータの標準格納方法がない）
- JSON .mbt の dtype 対応

## Considered Options

### A: float32 のみを維持する

- 利点: 変更なし
- 欠点: モデルサイズ削減の手段がない。ONNX f16 モデルのインポート不可

### B: 最初から演算まで f16/int8 をネイティブ対応する

- 利点: 推論時のメモリ効率最大化
- 欠点: MoonBit に Float16/Int8 型がないため、全演算カーネルをビット操作で再実装する必要がある。実装コストが非常に高い

### C: 外部ツールに dtype 変更を全面的に任せる

- 利点: MbTorch 側の変更なし
- 欠点: ラウンドトリップ不可。MbTorch で学習したモデルを小さくする手段がなくなる

## Consequences

### 利点

- Binary .mbt でモデルサイズ 50%（f16）〜 75%（int8）削減
- ONNX / safetensors の f16 モデルをインポート可能
- 既存の演算パスに影響なし
- 将来のネイティブ f16 演算への拡張パスが開ける

### 欠点

- Tensor 構造体に 2 フィールド追加（17 箇所の構築サイト変更）
- int8 は weight-only に限定（activation 量子化は将来課題）
- 演算結果の dtype が Float32 に戻る（明示的な to_dtype が必要）

## テスト戦略

19 テストケースを TDD で実装:

| テスト ID | 内容 |
|----------|------|
| D-T1〜T5 | float16 変換（往復、特殊値、精度損失、オーバーフロー、アンダーフロー）|
| D-T6〜T8 | int8 量子化（往復、自動レンジ、クランプ）|
| D-T9〜T10 | Tensor::to_dtype（Float16, Int8）|
| D-T11〜T14 | Binary .mbt（f16/int8 往復、後方互換、サイズ削減）|
| D-T15 | ONNX f16 テンソルパース |
| D-T16 | safetensors f16 ロード |
| D-T17 | ONNX f16 エクスポート往復 |
| D-T18 | MLP f16 推論パリティ |
| D-T19 | Linear int8 推論パリティ |

精度しきい値:
- f16 roundtrip: `|original - recovered| < 1e-3`
- int8 roundtrip: 相対誤差 < 5%
- MLP f16 推論: 絶対誤差 < 1e-2
- Linear int8 推論: 絶対誤差 < 0.5

## Links

- ADR-0005: JSON .mbt フォーマット
- ADR-0007: Phase 1 ONNX + safetensors
- ADR-0008: Binary .mbt フォーマット MVP
- ADR-0010: ONNX エクスポート MVP
