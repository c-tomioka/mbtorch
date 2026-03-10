# 0012: Self-Attention MVP — ONNX import/export 対応

* Status: accepted
* Date: 2026-03-10
* Deciders: @c-tomioka, Claude Code

## コンテキストと課題

MbTorch は MLP・CNN モデルの ONNX import/export を Phase 1-2 でサポートした。
Transformer 系モデルが主流となる中、Self-Attention の対応が次の重要ステップとなる。

ONNX には 2 つの Attention 表現がある:
- opset 23+ の `Attention` 演算子（統合 op）
- opset 13 以下の MatMul+Softmax サブグラフパターン（PyTorch の標準 export）

MoonBit にはネイティブの softmax や batched matmul がなく、
これらの基盤演算から構築する必要がある。

## 決定の判断基準

- 既存の sequential モデル API への最小限の変更
- ONNX opset 13 との互換性（最も広く使われるバージョン）
- autograd 対応（推論だけでなく学習も可能に）
- 将来の拡張余地（cross-attention、causal mask、LayerNorm）

## 決定内容

**MatMul+Softmax サブグラフパターンベースの Self-Attention MVP** を実装する。

1. `SelfAttention` struct を nn/ に追加（Wq/Wk/Wv/Wo の 4 つの Linear + num_heads）
2. Layer enum に `SelfAttention` variant を追加
3. core/tensor に `exp`, `softmax`, `batched_matmul_2d`, `transpose_last2` を追加
4. core/autograd に対応する 5 つの GradFn（Scale, Softmax, BatchedMatmul, TransposeLast2, Reshape）+ backward を追加
5. ONNX import: Gemm×3 + Transpose + MatMul + Div + Softmax + MatMul + Gemm パターンを検出
6. ONNX export: 同パターンのサブグラフとして展開（opset 13 互換）

### MVP スコープ

含める:
- Single-head / Multi-Head Self-Attention
- 入力 shape: [batch, seq_len, d_model] or [seq_len, d_model]
- マスクなし
- Float32 基本
- Autograd 対応（学習可能）

含めない:
- Cross-Attention
- Causal mask / padding mask
- KV キャッシュ
- ONNX Attention op (opset 23+)
- LayerNorm / TransformerBlock (別タスク)

## 検討した代替案

### A: Attention を一切サポートしない
- 利点: 変更なし
- 欠点: Transformer モデルの import 不可、MbTorch の実用性が制限される

### B: 最初から MHA + cross-attention + causal mask をフルサポート
- 利点: 一度で完全な Transformer 対応
- 欠点: 実装量が大きすぎる。mask の autograd、KV キャッシュの設計が複雑

### C: ONNX Attention op (opset 23+) のみサポート
- 利点: パターンマッチ不要、属性ベースで処理できる
- 欠点: opset 23 は新しく、多くの既存 ONNX モデルが opset 13-17。PyTorch の標準 export もサブグラフ

## 影響

### 利点
- Transformer 系モデル（BERT、GPT 等の小型版）の import/export が可能に
- softmax、batched matmul 等の基盤演算が他の用途にも利用可能
- autograd 対応により、Self-Attention を含むモデルの fine-tuning が可能

### 欠点
- core/autograd に 5 つの GradFn variant 追加（複雑度増加）
- 3D テンソル対応が batch loop ベース（将来的にはネイティブ対応が望ましい）
- ONNX import のパターンマッチが MbTorch 独自の export パターンに限定

## リンク

- ADR-0007: Phase 1 ONNX + safetensors
- ADR-0009: Conv2d / BatchNorm MVP
- ADR-0010: ONNX エクスポート MVP
- ADR-0011: DType 拡張（float16 / int8）
