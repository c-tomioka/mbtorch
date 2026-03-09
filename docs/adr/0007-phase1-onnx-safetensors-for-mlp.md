# 0007: Phase 1 — ONNX and safetensors support for small MLPs

* Status: accepted
* Date: 2026-03-09
* Deciders: @c-tomioka, Claude Code

## Context and Problem Statement

MbTorch はブラウザ・エッジ向けの軽量 ML フレームワークであり、外部で訓練済みのモデルをインポートする機能はロードマップの重要な柱である。ONNX は業界標準のモデル交換フォーマット、safetensors は Hugging Face エコシステムで広く使われる重みファイルフォーマットであり、どちらも対応が求められている。

現在 MbTorch には JSON ベースの `.mbt` I/O が実装済み（ADR-0005）で、Linear + ReLU ベースの MLP の保存・復元が動作している。Phase 1 として ONNX / safetensors の対応を開始するにあたり、どの範囲から着手し、どのような変換フローを採用するかを決める必要がある。

- 関連: ADR-0005 (JSON ベース .mbt フォーマット)
- 関連: ADR-0002 (レイヤードモジュール構成 — io は core, nn に依存可)

## Decision Drivers

- 小さな MLP（Linear + ReLU の数層）を最初のターゲットとし、スコープを限定したい
- Hugging Face エコシステム（safetensors）との連携を早期に確立したい
- 既存の .mbt JSON I/O を壊さず、拡張として追加したい
- MoonBit / WASM 環境でのバイナリパースの実装コストを考慮する必要がある
- ONNX の全 OP セットへの対応は Phase 1 では不要
- float32 / 2D テンソル以外の dtype・次元数への対応は後回しで構わない
- Phase 1 の成果物は「PyTorch → ONNX export → MbTorch import → 推論一致」の E2E パスとしたい

## Considered Options

- Option A: ONNX + safetensors を同時にミニマム対応（小さな MLP のみ）
- Option B: まず ONNX だけ import/export を実装し、safetensors は後回し
- Option C: まず safetensors ベースの weight import を実装し、ONNX Graph は後回し

## Decision Outcome

Chosen option: "Option A: ONNX + safetensors を同時にミニマム対応", because:

- 小さな MLP に絞れば、ONNX と safetensors の両方を同時に実装してもスコープが小さい（対応 OP: Gemm/MatMul, Add, Relu のみ）
- 実用的な変換フロー（ONNX Graph でネットワーク構造を読み、safetensors で重みを読む）を Phase 1 で一通り確立できる
- 片方だけ対応しても E2E の検証ができず、実用性を示せない
- ONNX 単体の重みデータは TensorProto にインラインで埋め込まれており、大きなモデルでは safetensors の方が扱いやすい
- safetensors はヘッダが JSON + 生バイト列という単純な構造で、パーサーの実装コストが低い
- 既存の io/ モジュールに `onnx.mbt` と `safetensors.mbt` を追加する形で、.mbt JSON I/O と共存できる

### Phase 1 の変換フロー（大枠）

```
[PyTorch Model]
    │
    ├── torch.onnx.export() ──→ model.onnx  (Graph 構造 + OP 情報)
    └── safetensors.save()  ──→ model.safetensors (重みデータ)

[MbTorch import]
    │
    ├── ONNX Graph をパース ──→ nn.Linear / ReLU の構成を再構築
    ├── safetensors をパース ──→ Tensor として重みデータを読み込み
    └── 結合 ──→ MbTorch Model (推論可能な状態)

[オプション: .mbt に再保存]
    └── 既存の serialize で .mbt JSON に変換可能
```

### Phase 1 で対応する範囲

| 項目 | 対応 |
|------|------|
| ONNX OP | Gemm (Linear), MatMul, Add, Relu のみ |
| safetensors dtype | float32 のみ |
| テンソル次元 | 1D, 2D のみ（weight/bias） |
| ネットワーク構造 | Sequential な MLP のみ（分岐・ループなし） |
| import (読み込み) | 対応 |
| export (書き出し) | Phase 2 以降 |

### Phase 1 で明示的にやらないこと

- Conv, BatchNorm, Attention, Pooling 等の OP 対応
- float16, int8, bfloat16 等の dtype 対応
- 3D 以上のテンソル（バッチ次元を含む動的形状など）
- ONNX Runtime との互換性検証
- ONNX export（MbTorch → ONNX 方向）
- safetensors への書き出し
- ONNX の動的形状（dynamic shapes）
- ONNX opset バージョン間の互換性処理

## Pros and Cons of the Options

### Option A: ONNX + safetensors 同時ミニマム対応（採用）

#### Pros
- E2E の変換フロー（PyTorch → MbTorch）を Phase 1 で確立できる
- 小さな MLP に絞れば実装量は manageable（ONNX パーサー + safetensors パーサー）
- 「構造（ONNX）+ 重み（safetensors）」の役割分担が明確で、拡張しやすい
- Hugging Face モデルの読み込みパスを早期に確保できる

#### Cons
- 2 つのフォーマットを同時に実装するため、Phase 1 の工数が Option B/C より大きい
- MoonBit でのバイナリパース（protobuf for ONNX, バイトレイアウト for safetensors）の実装が必要
- ONNX の protobuf パースは MoonBit エコシステムにライブラリがなく、手書きが必要 (assumption)

### Option B: ONNX のみ先行

#### Pros
- ONNX 単体でグラフ構造と重みの両方を含むため、1 ファイルで完結する
- 対応範囲が 1 フォーマットで済むため実装がシンプル

#### Cons
- ONNX TensorProto のインライン重みは大きなモデルでは非効率
- safetensors との連携が後回しになり、Hugging Face エコシステムとの接続が遅れる
- ONNX protobuf パーサーの実装コストだけでも相当量ある (assumption)

### Option C: safetensors のみ先行

#### Pros
- safetensors はフォーマットが単純（JSON ヘッダ + 生バイト列）で実装コストが最も低い
- Hugging Face エコシステムとの連携を最速で確立できる
- MoonBit のバイナリ操作で十分に実装可能

#### Cons
- safetensors には重みデータしか含まれず、ネットワーク構造の情報がない
- 構造情報を別途指定する仕組み（設定ファイル or コード記述）が必要になる
- E2E の「モデルファイルを渡すだけでインポート」体験を Phase 1 で提供できない

## Links

- Code: `io/types.mbt`, `io/serialize.mbt`, `io/deserialize.mbt` (既存 .mbt JSON I/O)
- ONNX spec: https://onnx.ai/onnx/intro/concepts.html
- safetensors spec: https://huggingface.co/docs/safetensors/
- Related ADRs: 0001 (MoonBit 言語選択), 0002 (レイヤードモジュール構成), 0005 (JSON ベース .mbt フォーマット)
