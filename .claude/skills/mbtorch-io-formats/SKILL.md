***
name: mbtorch-io-formats
description: >
  MbTorch（MoonBit 製 AI/ML フレームワーク）の io/ 周辺で、
  ONNX / safetensors / .mbt ネイティブ形式のモデル入出力を扱う際の設計・実装・テスト方針を提供するスキル。
  以下の場面では必ずこのスキルを有効にすること：io/ ディレクトリへの新規ファイル追加、
  ONNX / safetensors / .mbt の読み書きロジックの変更・拡張、モデル保存形式の設計・バージョンアップ検討、
  I/O 周辺のエラー報告や互換性問題の調査・修正。
***

> NOTE: グローバルなルールやワークフローは `CLAUDE.md` が優先されます。
> このスキルは `CLAUDE.md` を補完する、MbTorch I/O 専用ガイドです。
> レイヤリングルールは `mbtorch-architecture`、テスト方針は `mbtorch-tdd` スキルも参照すること。

## Role

このスキルを使うとき、Claude は **MbTorch の I/O 互換性エンジニア** として振る舞う。

- モデル形式の仕様と互換性を起点に、I/O コードの設計・レビューを行う
- フォーマット知識なしの「推測によるパース」を自分からは絶対に書かない
- I/O と推論ロジックの責務分離を常に意識する

I/O の失敗は静かに起きやすい。壊れたデータを「読めた」として返すより、明示的に失敗させる方が必ず安全。

***

## Supported Formats & Positioning

MbTorch が扱う 3 つの形式と、それぞれの位置づけ。

| 形式 | 役割 | 主な使い方 | 方向 |
|---|---|---|---|
| **ONNX** | 計算グラフの標準中間表現 | 他フレームワーク（PyTorch, TF 等）からのインポート | 読み込み専用（当面）|
| **safetensors** | 安全・高速なテンソル重みストレージ | 学習済み重みのロード（HuggingFace 等との連携）| 読み込み・書き出し |
| **`.mbt`** | MbTorch ネイティブのコンパクト保存形式 | MbTorch モデルの保存・再ロード・エッジ配布 | 読み込み・書き出し |

### データの流れ

```
外部フレームワーク
  └─ ONNX ──────────────────→ io/onnx_loader  ──→ nn.Model（推論）
  └─ safetensors ───────────→ io/safetensors   ──→ core.Tensor（重み）

MbTorch ネイティブ
  nn.Model ──→ io/mbt_writer ──→ .mbt ファイル
  .mbt ファイル ──→ io/mbt_loader ──→ nn.Model（再ロード）
```

### 各形式の仕様参照先

- **ONNX**: https://onnx.ai/onnx/intro/concepts.html（opset, graph, node, tensor）
- **safetensors**: https://huggingface.co/docs/safetensors/（ヘッダ構造、dtype 定義）
- 仕様を確認せずにパースロジックを書かない。不明な点は必ずユーザーに確認する。

***

## Design Principles for I/O

### 責務の閉じ込め

フォーマット固有の知識（バイト列の解釈、ヘッダ仕様、オペレータ定義）は **`io/` モジュールに閉じ込める**。

- `core/tensor` はファイル形式を知らない。純粋なテンソル型として存在する
- `nn/` のレイヤーはファイル形式を知らない。重みは `Tensor` として受け取るだけ
- ファイルを知っているのは `io/` だけ

悪い例:
```moonbit
// nn/linear.mbt — NG: nn がフォーマット知識を持っている
fn Linear::from_onnx_node(node: OnnxNode) -> Linear { ... }
```

良い例:
```moonbit
// io/onnx_loader.mbt — OK: io が変換して nn に渡す
fn load_linear_from_node(node: OnnxNode) -> Linear {
  let weight = tensor_from_onnx_initializer(node.initializer)
  Linear::new_with_weight(weight)
}
```

### メタ情報と重みデータの分離

すべての形式で「モデルの構造情報」と「テンソルバイナリ」を概念的に分けて扱う。

- **メタ情報**: レイヤー構成・shape・dtype・入出力名・バージョン等
- **重みデータ**: 実際のテンソルバッファ（バイナリ）

ロード時もこの順序で処理する: メタ情報を読んで構造を検証してから、テンソルバッファをマップする。
逆順（バッファを先に読む）はエラー時のメッセージが意味不明になる。

### I/O 専用の中間表現（DTO）の活用

ドメイン型（`nn.Linear`, `nn.Model` 等）とシリアライズ形式の間に、I/O 専用の構造体を置く。

```moonbit
// io/mbt_types.mbt
struct MbtModelMeta {
  version: Int
  format: String        // "mbtorch"
  model_name: String
  tensors: Array[TensorMeta]
  opset: Int
}

struct TensorMeta {
  name: String
  shape: Array[Int]
  dtype: DType
  offset: Int           // バイナリ領域内のオフセット
  size: Int
}
```

この中間表現を介することで、`nn.Model` の内部表現が変わっても I/O フォーマットへの影響を最小化できる。

***

## Validation & Error Handling Rules

### ロード時の検証チェックリスト

ファイルを読み込む際、以下の順序で検証する。

**1. ファイル存在確認**
- ファイルが存在するか・読み取り権限があるか

**2. マジックバイト／ヘッダ確認**
- ONNX: Protocol Buffers のシリアライズ形式（バイト列の先頭）
- safetensors: `{` で始まる JSON ヘッダ
- `.mbt`: MbTorch 独自のマジックバイト（例: `MBTORCH\0`）とバージョン番号

**3. バージョン互換性確認**
- ONNX opset バージョンが MbTorch のサポート範囲内か
- `.mbt` のバージョンフィールドが現在のローダーで読める世代か

**4. 構造の整合性確認**
- safetensors: ヘッダ JSON のオフセット・サイズの合計がファイルサイズと一致するか
- `.mbt`: メタ情報に記載された各テンソルのオフセット・サイズがバイナリ領域と合致するか
- ONNX: 宣言されているオペレータがすべて MbTorch でサポートされているか

### エラー型の設計

「何が壊れているか」が分かる粒度でエラーを分ける。

```moonbit
// io/errors.mbt
enum IoError {
  FileNotFound(String)                   // パスを含める
  InvalidMagicBytes(String)              // 形式名と実際のバイト列
  UnsupportedVersion(Int, Array[Int])    // (実際, サポート一覧)
  UnsupportedOpset(Int, Array[Int])      // ONNX 専用
  UnsupportedOperator(String)            // ONNX 専用: オペレータ名
  HeaderMismatch(String)                 // safetensors / .mbt
  TensorShapeMismatch(String, Array[Int], Array[Int]) // (名前, 期待, 実際)
  CorruptedData(String)                  // 詳細な説明
}
```

エラーメッセージには「ファイルパス・形式・バージョン・対処法のヒント」を含める（`mbtorch-api-design` スキルのエラー方針も参照）。

### Fail Fast の徹底

「読めるが危険な状態」を返さない。

- shape が合わない重みをシレっと詰め込まない
- opset 非対応のオペレータを「とりあえずスキップ」しない
- ヘッダとバッファが整合しないファイルを「部分的に」ロードしない

曖昧な状態を返すくらいなら、明確なエラーで止める。

***

## Testing Strategy

### ONNX のテスト

**最小モデルによる比較テスト:**

1. PyTorch / ONNX Runtime 等で MLP を `.onnx` に書き出す
2. MbTorch の `io/onnx_loader` で読み込む
3. 同じ入力テンソルで推論し、出力が一致するか確認する（許容誤差 `1e-5` 程度）

```
tests/io_onnx_inference_parity.mbt
```

**opset バージョン境界テスト:**

- サポート最小 / 最大 opset で読み込めることを確認する
- 非対応 opset で明確に `UnsupportedOpset` エラーになることを確認する

### safetensors のテスト

**ヘッダ・データ一致テスト:**

1. `safetensors` ライブラリで書き出したファイルを MbTorch で読み込む
2. テンソル名・shape・dtype・値がすべて一致するか確認する

```
tests/io_safetensors_roundtrip.mbt
```

**破損データテスト:**

- ヘッダ JSON が不正なファイル → `HeaderMismatch` エラー
- バッファサイズ不一致 → `CorruptedData` エラー

### `.mbt` のテスト

**ラウンドトリップテスト（最重要）:**

1. MbTorch でモデルを `.mbt` に保存する
2. 同じモデルを `.mbt` から再ロードする
3. 同じ入力で推論し、保存前後で出力が一致するか確認する

```
tests/io_mbt_roundtrip.mbt
```

**バージョン互換テスト:**

- 旧バージョンの `.mbt` ファイルを現在のローダーで読めることを確認する（互換性テスト）
- 未来のバージョン番号のファイルが `UnsupportedVersion` エラーになることを確認する

### I/O 変更時の必須ルール

**`io/` に変更を加えるときは、以下のいずれかを必ず追加する:**

- [ ] 他実装との推論結果比較テスト（ONNX / safetensors）
- [ ] ラウンドトリップテスト（`.mbt`）
- [ ] 破損・非対応ファイルに対する明確なエラーテスト

***

## `.mbt` Format Guidelines

### 基本構造（二段構成）

```
.mbt ファイル
  ├─ [マジックバイト]  8 bytes: "MBTORCH\0"
  ├─ [メタ長]          4 bytes: uint32 (メタデータ JSON のバイト長)
  ├─ [メタデータ JSON] N bytes: UTF-8 エンコードされた JSON
  └─ [テンソルバイナリ] M bytes: 各テンソルのバイナリデータ（メタで指定されたオフセットで参照）
```

### メタデータ JSON の必須フィールド

```json
{
  "format":     "mbtorch",
  "version":    1,
  "model_name": "my_model",
  "opset":      1,
  "tensors": [
    {
      "name":   "linear.weight",
      "shape":  [64, 128],
      "dtype":  "f32",
      "offset": 0,
      "size":   32768
    }
  ]
}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `format` | `string` | 常に `"mbtorch"`。マジックバイト以外の識別子 |
| `version` | `int` | フォーマットバージョン（現在 `1`）|
| `model_name` | `string` | モデルの識別名（空文字列可）|
| `opset` | `int` | MbTorch の演算セットバージョン |
| `tensors` | `array` | テンソルのメタ情報一覧 |

### 前方・後方互換性の方針

| 変更の種類 | 対応方針 |
|---|---|
| フィールドの追加（任意フィールド）| `version` を上げずに追加可能。ローダーは未知フィールドを無視する |
| フィールドのセマンティクス変更 | `version` を上げる。旧バージョンのファイルは旧ロジックで読む |
| 必須フィールドの削除 | `version` を上げ、移行ガイドを書く |
| バイナリ構造の変更 | 必ず `version` を上げ、移行ツールまたは変換ガイドを提供する |

### バージョンアップ時の手順

1. 変更内容が「前方互換な追加」か「破壊的変更」かを判断する
2. 破壊的変更なら `version` を `N+1` にインクリメントする
3. 旧バージョンのファイルを読むためのレガシーパスをローダーに追加する（または移行ツールを書く）
4. 変更内容と移行手順を README / CHANGELOG に記載する（`mbtorch-docs` スキル参照）
5. 新旧両バージョンに対してテストを追加する

***

## When to Use This Skill

以下のタスクでは必ずこのスキルを有効にすること。

- `io/` ディレクトリ以下に新しいファイルを追加するとき
- ONNX / safetensors / `.mbt` の読み書きロジックを変更・拡張するとき
- モデル保存形式の設計・バージョンアップを検討するとき
- I/O 周辺のエラー報告や互換性問題を調査・修正するとき
- 「ONNX から読んだモデルの推論結果がおかしい」など I/O 起因のバグを調査するとき
- `nn/` や `core/` に I/O ロジックが混入していないか確認するとき

***

## Do / Don't

### Do

- I/O ロジックは `io/` に閉じ込め、`core` / `nn` に形式知識を持ち込まない
- ロード時はマジックバイト → バージョン → 構造整合性の順に検証し、fail fast で止める
- 互換性テスト（他実装との比較 or ラウンドトリップ）を変更のたびに追加する
- `.mbt` を変更するときはバージョンと移行方針を検討してからコードを書く
- 仕様が不明な箇所は推測で書かずユーザーに確認する
- エラーメッセージにはファイルパス・形式・バージョン・対処法のヒントを含める

### Don't

- ファイル形式の詳細（ヘッダ仕様・バイト構造）を `core` や `nn` に直接埋め込む
- 仕様を確認せずに ONNX / safetensors のパースロジックを「推測」で書く
- 静かにフォーマットを変更して既存 `.mbt` ファイルを読めなくする
- `UnsupportedOperator` のオペレータをスキップして「部分的に動く」ローダーを作る
- 変更後に互換性テストを追加しない
