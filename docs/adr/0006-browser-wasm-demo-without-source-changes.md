# 0006: 初のブラウザデモは MbTorch 内部 MLP 学習を wasm-gc でそのまま動かす

* Status: accepted
* Date: 2026-03-08
* Deciders: @c-tomioka, Claude Code

## Context and Problem Statement

MbTorch は「ブラウザおよびエッジ環境での ML 推論・ファインチューニング」を主要ターゲットの一つとしている（ADR-0001）。Phase 1 MVP（Tensor / Autograd / NN / Optim / IO、テスト 78 件）の完成にあたり、CLI で動作する 2 層 MLP 訓練デモ（`examples/basic_mlp`）をブラウザでも動かせることを実証する必要があった。

論点は以下の 3 つに集約される:

- 初のブラウザデモのスコープをどこまでにするか（ONNX/safetensors インポートを含めるか）
- CLI 用コードとブラウザ用コードの関係をどう設計するか（共有 or 分離）
- ブラウザ側の出力方式と UI レベルをどこに設定するか

## Decision Drivers

- 「Write once, run anywhere（CLI/ブラウザ）」を最小コストで実証したい
- MoonBit の wasm-gc ターゲットの実用性を早期に検証したい
- ブラウザ用に別コードを保守するコストを避けたい
- ONNX / safetensors の IO 実装はまだブラウザ向けに十分に検証されていない
- wasm-gc の対応ブラウザに制限がある（Chrome 119+, Firefox 120+, Safari 18.2+）
- `use-js-builtin-string` は Safari など非対応ブラウザがあり、クロスブラウザ互換性を優先すべき
- 最初のデモは「動くこと」が最重要であり、リッチな UI は後回しでよい

## Considered Options

- Option A: CLI 専用デモだけに留め、ブラウザ対応は後回しにする
- Option B: 最初から ONNX / safetensors インポート込みのブラウザデモを目指す
- Option C: MbTorch 内部の 2 層 MLP 学習をそのまま wasm-gc でブラウザに持ち込み、コンソールログ中心の最小デモとする

## Decision Outcome

Chosen option: "Option C: MbTorch 内部 MLP 学習を wasm-gc でブラウザに持ち込む最小デモ", because:

- `examples/web_mlp/main.mbt` は `examples/basic_mlp/main.mbt` と同一のコアロジックで動作し、ソースコードを 1 行も変えずに CLI とブラウザの両方で実行できることを実証した
- `moon build --target wasm-gc` でビルドし、`_start` エクスポートを JS から呼び出すだけで動作する
- `println` は `spectest.print_char` を通じてブラウザ側に 1 文字ずつ渡し、JS 側でバッファリングしてコンソール出力する方式を採用した
- `use-js-builtin-string` は Safari 等の非対応ブラウザがあるため採用せず、`spectest.print_char` のみをインポートすることでクロスブラウザ互換性を確保した
- エントリポイントを別パッケージ（`examples/web_mlp`）に分けつつ、コアロジックは `core`, `nn`, `optim` の共有ライブラリをそのまま使う構成とした
- UI は最小限（HTML/JS の簡単なメッセージ表示）に留め、後続フェーズでリッチ化する余地を残した
- @c-tomioka により Chrome, Safari, Firefox, Brave の 4 ブラウザで動作確認済み

## Pros and Cons of the Options

### Option A: CLI 専用デモのみ、ブラウザ対応は後回し

#### Pros
- 実装コストが最小
- wasm-gc のクロスブラウザ差異を気にしなくてよい

#### Cons
- MbTorch の主要価値提案（ブラウザ/エッジ ML）を実証できない
- MoonBit の WASM-first 言語選択（ADR-0001）の意義が薄れる
- ブラウザ対応が先送りされるほど技術的リスクの発見が遅れる

### Option B: ONNX / safetensors インポート込みのブラウザデモ

#### Pros
- 実用的なモデル読み込みワークフローを一気に実証できる
- ユーザーに対するインパクトが大きい

#### Cons
- IO 層のブラウザ対応（ファイル読み込み、バイナリ解析の WASM 互換性）が未検証で、技術リスクが高い
- スコープが大きくなり、初回デモとしてのリードタイムが長くなる
- 失敗時の切り分けが困難（WASM 問題なのか IO 問題なのか判別しにくい）

### Option C: MbTorch 内部 MLP 学習をそのまま wasm-gc で最小デモ（採用）

#### Pros
- ソースコードの二重管理が不要（CLI と同一ロジック）
- MoonBit の WASM-first 設計を最も直接的に実証できる
- `spectest.print_char` のみで動作するためインポートが 1 つだけで、クロスブラウザ互換性が高い
- デモの失敗要因が限定的で、問題の切り分けが容易
- 後続フェーズ（ONNX インポート、リッチ UI）への段階的拡張が自然にできる

#### Cons
- ブラウザ固有の UI 操作（DOM 操作、グラフ描画等）は JS 側に委譲する必要がある
- wasm-gc 対応ブラウザに制限がある（ただし主要 4 ブラウザで動作確認済み）
- 外部モデルの読み込みは含まれないため、実用性のアピールとしては限定的

## Links

- Commit: `a28883b` CLI およびブラウザでもMLP訓練デモを実装
- Code: `examples/web_mlp/main.mbt`
- Code: `examples/web_mlp/index.html`
- Code: `examples/web_mlp/README.md`
- Code: `examples/basic_mlp/main.mbt`
- Related ADRs: 0001 (MoonBit 言語選択), 0002 (レイヤードモジュール構成), 0005 (JSON ベース .mbt フォーマット)
