# 0006: ブラウザ/WASM デモをソースコード変更なしで実現

* Status: accepted
* Date: 2026-03-08
* Deciders: @c-tomioka, Claude Code

## Context and Problem Statement

MbTorch は「ブラウザおよびエッジ環境での ML 推論・ファインチューニング」を主要ターゲットの一つとしている。CLI で動作する訓練デモ（`examples/basic_mlp`）をブラウザでも動かせることを実証する必要があった。

その際、ブラウザ用に別のソースコードを書くのか、同一のコードをそのまま WASM にコンパイルして動かせるのかが論点となった。

## Decision Drivers

- 「Write once, run anywhere（CLI/ブラウザ）」を実証したい
- MoonBit の wasm-gc ターゲットの実用性を検証したい
- ブラウザ用に別コードを保守するコストを避けたい
- ユーザーに WASM デプロイの手軽さをアピールしたい
- 訓練結果をブラウザ上で可視化できるようにしたい

## Considered Options

- 同一 MoonBit コードを wasm-gc ターゲットでコンパイルし、JS グルーコードで連携
- ブラウザ専用のコードを別途作成
- CLI デモのみ提供し、ブラウザデモは後回し

## Decision Outcome

Chosen option: "同一 MoonBit コードを wasm-gc ターゲットでコンパイルし、JS グルーコードで連携", because:

- `examples/web_mlp/main.mbt` は `examples/basic_mlp/main.mbt` とほぼ同一のロジックで動作する
- `moon build --target wasm-gc` でブラウザ用バイナリを生成できる
- `index.html` + 薄い JS レイヤーで訓練の進捗をリアルタイム表示できる
- MoonBit の WASM-first 設計（ADR-0001）の実用性を証明できた
- ソースコードの二重管理を避けられる
- @c-tomioka により Chrome, Safari, Firefox, Brave の 4 ブラウザで MLP 訓練デモの動作を確認済み

## Pros and Cons of the Options

### Option 1: 同一コード ＋ wasm-gc ＋ JS グルーコード

#### Pros
- ソースコードの二重管理が不要
- MoonBit の WASM-first を最大限活用
- CLI とブラウザで動作の一貫性が保証される
- 小さなバイナリサイズでブラウザ配信に適している (assumption based on git history)

#### Cons
- ブラウザ固有の UI 操作（DOM 操作等）は JS 側に委譲する必要がある
- wasm-gc の対応ブラウザに制限がある（ただし Chrome, Safari, Firefox, Brave で動作確認済み）

### Option 2: ブラウザ専用コードを別途作成

#### Pros
- ブラウザ固有の最適化やUI統合が自由にできる

#### Cons
- ソースコードの二重管理が発生する
- CLI 版とブラウザ版の動作差異が生じるリスク
- 保守コストが倍増する

### Option 3: CLI デモのみ

#### Pros
- 実装コストが最小

#### Cons
- MbTorch の主要価値提案（ブラウザ/エッジ ML）を実証できない
- WASM-first の言語選択（ADR-0001）の意義が薄れる

## Links

- Commit: `a28883b` CLI およびブラウザでもMLP訓練デモを実装
- Code: `examples/web_mlp/main.mbt`
- Code: `examples/web_mlp/index.html`
- Code: `examples/basic_mlp/main.mbt`
- Related ADRs: 0001, 0002
