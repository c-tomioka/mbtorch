***
name: mbtorch-examples
description: >
  MbTorch（MoonBit 製 AI/ML フレームワーク）の examples / デモを設計・実装・更新するためのスキル。
  新しい API や機能を追加したときに「動く最小の例」を提案し、examples/ と API を常に同期させる。
  以下の場面では必ずこのスキルを有効にすること：新しい機能・API を追加してその使い方を示す例がまだないとき、
  既存 example を整理・分割したいとき、ブラウザ/エッジ向けデモのアイデアを試したいとき、
  README に載せるサンプルコードを考えるとき、API 変更後に examples/ を更新するとき。
***

> NOTE: グローバルなルールやワークフローは `CLAUDE.md` が優先されます。
> このスキルは `CLAUDE.md` を補完する、MbTorch examples 専用ガイドです。

## Role

このスキルを使うとき、Claude は **MbTorch の example / demo 作者** として振る舞う。

- 実装と同時に「ユーザーにどう見せるか」を考える
- 新しい API を追加するたびに「この機能を最小のコードで表現するとどうなるか」を自問する
- examples/ が API の生きたドキュメントとして機能するよう、常に動く状態に保つ

examples の価値は「動く」こと。読めるだけで実行できない例は、無いのと同じ。

***

## Example Design Principles

### 1 example = 1 ユースケース

- 1 つのディレクトリ・ファイルが伝えることを 1 つに絞る
- 「何でも入り」のデモは作らない。複数の概念を見せたいなら例を分割する
- タイトルを見て「何をする例か」が 5 秒で分かるように命名する

### 小さく、すぐ動く

- コードは数十行〜百行程度が理想。それを超えるなら分割を検討する
- セットアップが重くなる外部データ・外部 API への依存は避け、ハードコードか疑似データを使う
- `moon` コマンド 1〜2 行で実行できることを目指す

### 公開 API のみを使う

- `internal_` / private な型・関数への依存禁止
- 未安定（`@unstable` などのマーク）の API を使う場合は README に明記する

### 段階的な難易度を意識する

MbTorch の examples は以下のような「難易度の階段」を意識して設計する:

```
Hello, MbTorch（最小 MLP）
  ↓
CLI 訓練スクリプト
  ↓
ONNX インポートと推論
  ↓
ブラウザ / WASM デモ
  ↓
LoRA 等の軽量 fine-tuning
```

新しい example を追加するとき、この階段のどこに位置するかを考える。

***

## Example Types

### CLI examples（ローカル実行）

ローカルで `moon run` して動かすスクリプト。

**用途**: 訓練・推論・モデル I/O の基本動作確認
**ファイル構成**:
```
examples/<name>/
  main.mbt       # エントリポイント
  moon.pkg       # パッケージ設定
  README.md      # 実行手順と何を示すかの説明
```

**README に必ず書くこと**:
- 何をする例か（1〜2 文）
- 実行コマンド（`moon run` or `moon test`）
- 期待される出力

### Browser / WASM demos

WebAssembly にコンパイルし、ブラウザ上で動かすデモ。

**用途**: MbTorch の「ブラウザで動く ML」という特徴を伝える
**ファイル構成**:
```
examples/<name>/
  lib.mbt        # WASM にコンパイルされるコア処理
  moon.pkg
  index.html     # ブラウザ UI（最小限の HTML/JS）
  README.md      # ビルド手順・ブラウザでの確認方法
```

**README に必ず書くこと**:
- WASM ビルドコマンド（`moon build --target wasm32-unknown-unknown`）
- ブラウザで開く手順（ローカルサーバーが必要なら手順を書く）

### Edge-like demos（エッジ環境シミュレーション）

エッジデバイス環境を想定した軽量推論・センサーデータ処理のシミュレーション。

**用途**: リソース制約環境での動作を示す
**ファイル構成**: CLI examples と同じ構成に `edge_sim/` のようなサフィックスを付けて識別
**README に必ず書くこと**:
- 想定するエッジ環境（メモリ上限・CPU 制約のイメージ）
- モデルサイズと推論速度の概算（分かる場合）

***

## `examples/` Directory & Naming Rules

### ディレクトリ構成

```
examples/
  basic_mlp/          # 最小の MLP 訓練（入門）
  onnx_import/        # ONNX モデルの読み込みと推論
  safetensors_load/   # safetensors からの重み読み込み
  browser_mnist/      # WASM を使ったブラウザ上の MNIST 推論
  fine_tune_lora/     # LoRA を使った軽量 fine-tuning
  edge_inference/     # エッジ環境を想定した推論シミュレーション
```

### 命名ルール

- **ディレクトリ名**: `<機能>_<ユースケース>` の形で snake_case
  - 良い例: `browser_mnist`, `onnx_import`, `fine_tune_lora`
  - 悪い例: `demo1`, `test_example`, `misc`
- **メインファイル**: `main.mbt`（CLI / エッジ）または `lib.mbt`（WASM エクスポート用）
- **テストファイル**: `main_test.mbt`（ある場合）

### 各ディレクトリに必ず含めるもの

| ファイル | 必須 | 内容 |
|---|---|---|
| `main.mbt` / `lib.mbt` | 必須 | メインコード |
| `moon.pkg` | 必須 | パッケージ設定 |
| `README.md` | 必須 | 実行手順・何を示す例か |

***

## Example Implementation Guidelines

### コードを書くときのルール

- `moon` コマンドで実行できる形にする（`moon run`, `moon test`, `moon build`）
- エラー処理は最小限にし、ハッピーパスに集中する（エラーハンドリングのデモでない限り）
- ログ出力は `println` 等を使い、進捗や結果が見える形にする
- コメントは「何をしているか」ではなく「なぜそうするか」を書く

### データの扱い

- 外部データセット（MNIST 等）を必要とする例は、セットアップ手順を README に明示する
- 可能なら疑似データ（ランダム生成 or ハードコード定数）でも動く設計にする

  ```moonbit
  // Instead of loading real data, generate synthetic data
  let x = Tensor::randn([32, 128])  // batch_size=32, features=128
  let y = Tensor::randint(0, 10, [32])  // 10-class labels
  ```

### テストとの連携

可能な場合、example に最低限のテストを付ける:

- ビルドが通るか（コンパイルエラーがないか）
- 訓練ループが実行できるか（数ステップだけでよい）
- 推論が出力を返すか（精度チェックは不要、形状チェックだけでよい）

CI や定期チェックで「examples がビルドできるか」を確認する体制を将来的に整える方針。

***

## Example & API Synchronization

### 新しい API を追加したとき

1. その API を使う最小の example が既にあるか確認する
2. なければ、既存の example に 1〜2 行追加するか、新しい example を作るかを検討する
3. README のクイックスタートコードもその API を使う形に更新することを提案する

### 既存 API を変更したとき

1. `examples/` を検索し、変更した API を使っているファイルを探す
2. 見つかった箇所を新しいシグネチャに合わせて更新する
3. README のサンプルコードも同様に確認・更新する

この 2 ステップを「コード変更後のチェックリスト」として習慣化する。
（`mbtorch-docs` スキルの Update Strategy とセットで使うと効果的）

### README との連携

- README のクイックスタートコードは、`examples/basic_mlp/` の実ファイルに対応させる
- 長いコード例は README に直接書かず、`examples/` へのリンクにとどめる

  ```markdown
  See [examples/browser_mnist/](examples/browser_mnist/) for a full browser demo.
  ```

***

## When to Use This Skill

以下のタスクでは必ずこのスキルを有効にすること。

- 新しい機能・API を追加して、その使い方を示す example がまだないとき
- 既存の example が肥大化していて、分割・整理したいとき
- ブラウザ / エッジ向けの新しいデモアイデアを試したいとき
- README に載せるサンプルコードを設計するとき（`mbtorch-api-design` と併用可）
- API 変更後に `examples/` の影響箇所を探して更新するとき
- 「この機能の使い方が分かる example を作ってほしい」と言われたとき

***

## Do / Don't

### Do

- まず「この機能を説明するのに最小限のコードは何行か？」を考えてから書き始める
- 例は動くものにし、実行手順を README に必ず書く
- MbTorch の特徴（MoonBit, WASM, エッジ推論, 軽量 fine-tuning）が伝わる例を優先する
- API を変更したら `examples/` を検索して影響箇所を更新する
- 疑似データでも動く形にしてセットアップの負担を下げる

### Don't

- 1 つの example に複数の概念・API を詰め込みすぎる（1 example = 1 ユースケース）
- private / internal API に依存する例を書く
- 実行方法が分からない example（README なし、コマンド例なし）を追加する
- セットアップが重くて「動かすまでに 10 分以上かかる」例を作る（外部データは README で案内するにとどめる）
- examples/ を更新せずに API だけ変える
