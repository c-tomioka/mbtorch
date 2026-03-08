# 0005: JSON ベース .mbt フォーマットによるモデル I/O

* Status: accepted
* Date: 2026-03-08
* Deciders: @c-tomioka, Claude Code

## Context and Problem Statement

MbTorch で訓練したモデルを保存・読み込みするためのフォーマットを決める必要があった。ONNX や safetensors といった既存フォーマットのサポートはロードマップに含まれているが、MbTorch 固有のネイティブフォーマット（`.mbt`）をどのような形式で最初に実装するかが論点となった。

初期段階ではデバッグのしやすさと実装の速さを優先し、将来的にはバイナリフォーマットへの移行も視野に入れる必要がある。

## Decision Drivers

- 初期実装のコストを低く抑えたい
- 人間が読める形式でデバッグを容易にしたい
- ラウンドトリップ（serialize → deserialize）の正確性を検証しやすくしたい
- MoonBit の標準ライブラリで扱える形式にしたい
- 将来的なバイナリフォーマットへの移行パスを残したい

## Considered Options

- JSON ベースの .mbt フォーマットを最初に実装し、バイナリは後から追加
- 最初からバイナリの .mbt フォーマットを実装
- カスタムフォーマットは作らず、ONNX / safetensors のみを使用

## Decision Outcome

Chosen option: "JSON ベースの .mbt フォーマットを最初に実装", because:

- JSON は人間が読めるため、シリアライズ結果の目視確認やデバッグが容易
- MoonBit の標準ライブラリで JSON の生成・パースが可能
- DTO（TensorData, LayerData, MbtModel）を定義し、構造化された形式で保存できる
- `format` と `version` フィールドにより、将来のフォーマット変更に備えられる
- テストでラウンドトリップの正確性を容易に検証できる

## Pros and Cons of the Options

### Option 1: JSON ベース .mbt（初期実装）→ バイナリ（将来）

#### Pros
- 実装コストが低い
- デバッグが容易（テキストエディタで中身を確認できる）
- テストが書きやすい（文字列比較で検証可能）
- バージョニングにより将来の移行が安全

#### Cons
- ファイルサイズがバイナリより大きい
- 浮動小数点数のテキスト表現で精度が落ちる可能性 (assumption)
- パース速度がバイナリより遅い (assumption)

### Option 2: 最初からバイナリ .mbt

#### Pros
- ファイルサイズが小さい
- ロード速度が速い (assumption)
- 浮動小数点数の精度を保てる

#### Cons
- 実装コストが高い
- デバッグが困難（バイナリビューアが必要）
- MoonBit でのバイナリ操作の成熟度が不明 (assumption)

### Option 3: ONNX / safetensors のみ

#### Pros
- 既存エコシステムとの互換性が高い
- 他のフレームワークとのモデル交換が容易

#### Cons
- ONNX パーサーの実装コストが高い
- MbTorch 固有の情報（レイヤー構成等）を表現しづらい (assumption)
- 初期段階では過剰な複雑さを持ち込む

## Links

- Commit: `ea27f80` io モジュール実装
- Code: `io/types.mbt` (TensorData, LayerData, MbtModel DTO)
- Code: `io/serialize.mbt`
- Code: `io/deserialize.mbt`
- Code: `io/io_test.mbt`
- Related ADRs: 0001, 0002
