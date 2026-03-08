# 0001: MoonBit を実装言語として採用

* Status: accepted
* Date: 2026-03-07
* Deciders: @c-tomioka, Claude Code

## Context and Problem Statement

MbTorch は軽量な AI/ML フレームワークとして、WebAssembly (WASM) およびエッジ環境での推論・軽量ファインチューニングを主要ターゲットとしている。この目標を達成するために、どのプログラミング言語で実装するかを決定する必要があった。

WASM への効率的なコンパイル、小さなバイナリサイズ、高速なコンパイル時間、そして AI/ML ワークロードへの適性が求められた。

## Decision Drivers

- WASM ファーストであること（ブラウザ・エッジでネイティブに動作する）
- コンパイル後のバイナリサイズが小さいこと
- コンパイル速度が高速であること
- 型安全性が高く、バグを早期に検出できること
- AI/ML 向けの数値計算が自然に書けること

## Considered Options

- MoonBit を採用する
- Rust + wasm-bindgen を採用する
- C/C++ + Emscripten を採用する
- AssemblyScript (TypeScript サブセット) を採用する

## Decision Outcome

Chosen option: "MoonBit を採用する", because:

- WASM をファーストクラスターゲットとして設計されており、WASM 向けの最適化が言語レベルで行われている
- コンパイル後のバイナリサイズが非常に小さく、エッジ・ブラウザ環境に適している
- コンパイル速度が高速で、TDD サイクルを素早く回せる
- 強い型システムにより数値計算のバグを早期検出できる
- パッケージシステムが整備されており、モジュラーな設計が容易である

## Pros and Cons of the Options

### Option 1: MoonBit

#### Pros
- WASM ファーストの言語設計で、wasm-gc ターゲットに最適化されている
- バイナリサイズが非常に小さい
- コンパイルが高速（TDD に有利）
- 強い型推論と代数的データ型

#### Cons
- エコシステムが発展途上であり、サードパーティライブラリが少ない (assumption based on git history)
- コミュニティが小さく、情報リソースが限られる (assumption based on git history)
- 言語仕様が変化する可能性がある (assumption based on git history)

### Option 2: Rust + wasm-bindgen

#### Pros
- 成熟したエコシステムと豊富なライブラリ
- メモリ安全性の保証
- WASM サポートが充実

#### Cons
- コンパイル時間が長く、TDD サイクルが遅くなる (assumption)
- WASM バイナリサイズが MoonBit と比較して大きくなりがち (assumption)
- 学習コストが高い

### Option 3: C/C++ + Emscripten

#### Pros
- 既存の ML ライブラリ資産を流用できる可能性
- 高い実行パフォーマンス

#### Cons
- メモリ安全性の問題
- Emscripten によるビルド設定が複雑 (assumption)
- WASM バイナリサイズが大きくなりやすい (assumption)

### Option 4: AssemblyScript

#### Pros
- TypeScript に近い構文で学習コストが低い
- WASM ネイティブ

#### Cons
- 型システムが弱く、ML フレームワークの型安全性に不安 (assumption)
- パフォーマンスが MoonBit や Rust に劣る可能性 (assumption)

## Links

- Commit: `ac2baf1` ベースコード作成
- Commit: `10bf88c` カスタムインストラクション追加
- Code: `.claude/CLAUDE.md` (Tech stack & scope セクション)
- Code: `moon.mod.json`
