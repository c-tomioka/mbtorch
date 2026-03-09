# 0008: バイナリ .mbt フォーマット MVP

* Status: proposed
* Date: 2026-03-09
* Deciders: @c-tomioka, Claude Code

## Context and Problem Statement

ADR-0005 で JSON ベースの .mbt フォーマットを実装した際、「将来的にはバイナリフォーマットへの移行パスを残す」ことを明示的に記載していた。JSON .mbt は人間可読でデバッグに便利だが、ファイルサイズが大きく（float64 テキスト表現）、パース速度もバイナリに劣る。

ブラウザ・エッジ環境ではモデルのダウンロード・ロード速度が重要であり、float32 バイナリ形式への対応が求められている。safetensors パーサーの実装（ADR-0007）で MoonBit でのバイナリ操作が十分に成熟していることが確認できた。

## Decision Drivers

- JSON .mbt との相互変換を可能にしたい（同一モデルを両形式で保存・読込）
- safetensors パーサーで実証済みのバイナリユーティリティ（`read_u32_le`, `float32_le_to_double`）を再利用したい
- MVP では float32 / 1D・2D テンソル / Linear 層のみをスコープとする
- 既存の JSON .mbt API を変更せず、新しい API として追加する
- ファイルサイズを JSON .mbt の 1/4〜1/8 に削減したい

## Considered Options

- JSON メタデータ + バイナリテンソルバッファのハイブリッド形式（safetensors 方式）
- 完全バイナリ形式（メタデータも含めてすべてバイナリ）
- JSON .mbt にバイナリテンソルを Base64 エンコードで埋め込む
- safetensors フォーマットをそのまま採用

## Decision Outcome

Chosen option: "JSON メタデータ + バイナリテンソルバッファのハイブリッド形式", because:

- safetensors パーサーの実装経験から、このアプローチの有効性が実証済み
- JSON メタデータにより、テンソルのオフセット・形状をヘッダから読み取れる
- float32 バイナリにより、JSON .mbt の 1/4〜1/8 のファイルサイズを実現
- 既存 API と共存する新規 API として追加するため、後方互換性の問題がない

### ファイル構造

```
Offset   Size      Field           Description
───────────────────────────────────────────────────
0        4 bytes   magic           ASCII "MBTM" (0x4D 0x42 0x54 0x4D)
4        4 bytes   version         u32 LE (value: 1)
8        4 bytes   metadata_size   u32 LE (JSON メタデータのバイト長)
12       N bytes   metadata        UTF-8 JSON 文字列
12+N     M bytes   tensor_buffer   float32 LE のテンソルデータ連結
```

### 新規 API

```
serialize_model_to_binary(layers: Array[(String, Linear)]) -> Bytes
deserialize_model_from_binary(data: Bytes) -> Array[(String, Linear)] raise MbtFormatError
```

### MVP スコープ

含める:
- float32 のみ
- 1D / 2D テンソルのみ
- Linear 層のみ
- シリアライズ / デシリアライズ双方向
- 既存 MbtFormatError による検証（magic, version, format, shape）

含めない:
- float16 / bfloat16 / int8 等の dtype 対応
- 3D 以上のテンソル対応
- Conv / BatchNorm / Attention 等の層タイプ対応
- 圧縮（gzip 等）
- チェックサム / 整合性検証

## Pros and Cons of the Options

### Option 1: JSON メタデータ + バイナリバッファ（採用）

#### Pros
- safetensors 実装の知見を再利用できる
- メタデータが人間可読なためデバッグがある程度可能
- ファイルサイズ削減（パラメータ部分が 1/4〜1/8）
- JSON .mbt との相互変換が容易

#### Cons
- float64 → float32 の精度低下が避けられない
- メタデータ部分は依然テキスト（ただしモデルサイズに対して無視できるほど小さい）

### Option 2: 完全バイナリ形式

#### Pros
- 最小ファイルサイズ
- パース速度が最速

#### Cons
- 独自バイナリエンコーディングの実装コストが高い
- デバッグが困難
- 将来のフォーマット拡張が難しい

### Option 3: JSON + Base64 埋め込み

#### Pros
- 単一の JSON ファイルで完結
- 既存の JSON パーサーで読める

#### Cons
- Base64 エンコードで 33% サイズ増加
- エンコード/デコードのオーバーヘッド
- 根本的な問題（ファイルサイズ）の解決にならない

### Option 4: safetensors をそのまま採用

#### Pros
- PyTorch / Hugging Face エコシステムとの互換性

#### Cons
- MbTorch 固有のレイヤー構造情報を持てない
- ONNX との併用が必要になる（構造 + 重みの分離）
- MbTorch ネイティブフォーマットとしての独立性がない

## Links

- ADR-0005: JSON ベース .mbt フォーマット
- ADR-0007: Phase 1 ONNX + safetensors
- Code: `io/binary.mbt` (実装)
- Code: `io/binary_test.mbt` (テスト)
- Code: `io/safetensors.mbt` (バイナリユーティリティの参照実装)
