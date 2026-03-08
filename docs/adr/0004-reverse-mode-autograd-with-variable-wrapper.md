# 0004: 逆伝播 autograd と明示的 Variable ラッパーの採用

* Status: accepted
* Date: 2026-03-08
* Deciders: @c-tomioka, Claude Code

## Context and Problem Statement

MbTorch でニューラルネットワークを訓練するには、損失関数から各パラメータへの勾配を自動計算する仕組み（自動微分）が必要である。どのような自動微分の設計を採用するかを決定する必要があった。

Tensor モジュールはすでに forward-only の数値演算を提供しており、autograd はこれを拡張して勾配計算を可能にする位置づけである。

## Decision Drivers

- PyTorch ユーザーにとって馴染みのある API にしたい
- 計算グラフの構築と勾配計算を明確に分離したい
- Tensor (forward-only) と Variable (autograd-aware) の責務を分けたい
- MoonBit の代数的データ型（enum）を活かした設計にしたい
- 勾配の正しさを数値的に検証できる仕組みを組み込みたい

## Considered Options

- Variable ラッパー ＋ GradFn enum による逆伝播（PyTorch 風）
- Tensor に直接勾配機能を組み込む
- 順伝播自動微分（Forward-mode AD）

## Decision Outcome

Chosen option: "Variable ラッパー ＋ GradFn enum による逆伝播", because:

- `Variable` が `Tensor` をラップし、`grad` と `grad_fn` を保持する明確な設計
- `GradFn` を MoonBit の enum として定義することで、演算の種類をパターンマッチで安全に処理できる
- `backward()` はスカラー Variable に対して呼び出し、トポロジカルソートで勾配を伝播する
- `grad_check` モジュールにより、有限差分法で勾配の正確性を自動検証できる
- `core/tensor` は autograd を知らないまま維持でき、レイヤリングルール（ADR-0002）を遵守できる

## Pros and Cons of the Options

### Option 1: Variable ラッパー ＋ GradFn enum（PyTorch 風）

#### Pros
- Tensor と Variable の責務が明確に分離される
- MoonBit の enum + パターンマッチと相性が良い
- PyTorch ユーザーに馴染みのある API
- `requires_grad` フラグで勾配計算の有無を制御できる

#### Cons
- Variable と Tensor の二重構造により、API 表面が増える
- 演算ごとに GradFn のバリアントを追加する必要がある

### Option 2: Tensor に直接勾配機能を組み込む

#### Pros
- API がシンプル（型が 1 つで済む）

#### Cons
- `core/tensor` が autograd の責務を持ち、レイヤリングルールに違反する
- 勾配不要な場面でもオーバーヘッドが発生する (assumption)
- 責務の肥大化により保守が困難になる

### Option 3: 順伝播自動微分

#### Pros
- 実装が比較的単純
- メモリ使用量が少ない (assumption)

#### Cons
- パラメータ数が多い場合に計算コストが高い
- ニューラルネットワークの訓練には逆伝播が標準的であり、既存知見との整合性が低い

## Links

- Commit: `804ac68` core/autograd モジュール実装
- Code: `core/autograd/types.mbt` (Variable, GradFn 定義)
- Code: `core/autograd/engine.mbt` (backward 実装)
- Code: `core/autograd/ops.mbt` (微分可能な演算)
- Code: `core/autograd/grad_check.mbt` (数値勾配検証)
- Related ADRs: 0001, 0002
