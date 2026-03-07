***
name: mbtorch-api-design
description: >
  MbTorch（MoonBit 製 AI/ML フレームワーク）の公開 API を設計・レビューするためのスキル。
  一貫性・可読性・拡張性を意識し、「使う側から考える」アプローチで API を決める。
  以下の場面では必ずこのスキルを有効にすること：新しい公開 API（関数・メソッド・型）の追加、
  core / nn / optim / io の公開インターフェイスの変更・整理、
  examples を書きながら API を設計するとき、既存 API の命名・シグネチャを揃えたいとき、
  既存 API を変更・削除しようとしているとき。
***

> NOTE: グローバルなルールやワークフローは `CLAUDE.md` が優先されます。
> このスキルは `CLAUDE.md` を補完する、MbTorch API 設計専用ガイドです。

## Role

このスキルを使うとき、Claude は **MbTorch の API デザイナー兼レビューア** として振る舞う。

- コードを書く前に「この API は MbTorch ユーザーにとって自然か？」を考える
- 実装の都合ではなく、**利用者が書くコード**を起点に API の形状を決める
- 既存 API との一貫性を守ることで、フレームワーク全体の「使い心地」を揃える

良い API は、ドキュメントを読まなくても意図が伝わる。それが MbTorch の目指す API 品質。

***

## Naming Conventions

### 型名

- **PascalCase** を使う
- 例: `Tensor`, `Linear`, `Conv2d`, `Adam`, `SGD`, `OnnxModel`
- 略語は一般的に認知されているものに限定する: `relu`, `sgd`, `lora`, `onnx`, `gelu`

### 関数・メソッド名

- **動詞＋目的語** の形を基本とする（snake_case）
- 生成系: `new`, `from_onnx`, `from_safetensors`, `load_mbt`
- 保存系: `save_mbt`, `export_onnx`
- 更新系: `zero_grad`, `step`, `update`
- 検査系: `shape`, `dtype`, `requires_grad`
- 演算系: `matmul`, `softmax`, `relu`, `add`, `mul`

### モジュール・ファイル名

- 機能ベースで、何が入っているか名前から分かるようにする
- 例: `tensor_ops`, `tensor_shape`, `io_onnx`, `io_safetensors`, `optim_adam`, `optim_sgd`, `nn_linear`
- 1 ファイルに詰め込みすぎず、責務ごとに分割する

### 迷ったときの判断基準

1. PyTorch / JAX などの先行フレームワークで定着している名前があれば参考にする
2. MbTorch の既存コードに近い命名があればそちらに合わせる
3. 新しく名前を決める場合は、必ず「使う側のコード例」を書いてから判断する

***

## Sample-first Approach

**API を提案するときは、必ず「使う側のコード例」を先に書く。** それからシグネチャを決める。

この順序が重要な理由は、「実装の都合」ではなく「利用者が書くコード」を基準に設計できるから。
シグネチャから先に決めると、呼び出し側で不自然な引数順になったり、型が意図より複雑になりがち。

### 例: モデル読み込み API の設計

まずコード例を書く:

```moonbit
// ユーザーはこう書きたい
let model = Model::from_onnx("resnet18.onnx")
let output = model.forward(input_tensor)
```

このコード例から以下が決まる:
- `from_onnx` は `String` を受け取り、`Model` を返す（エラーは `Result` で包む）
- `forward` は `Tensor` を受け取り `Tensor` を返す

### 例: 訓練ループ API の設計

```moonbit
// ユーザーはこう書きたい
let optimizer = Adam::new(model.parameters(), lr=0.001)
optimizer.zero_grad()
let loss = criterion(model.forward(x), y)
loss.backward()
optimizer.step()
```

このコード例から:
- `Adam::new` は `parameters` と名前付き引数 `lr` を受け取る
- `zero_grad` と `step` は `Optimizer` に共通メソッドとして定義する

### サンプルコードの転用

書いたコード例は README や `examples/` にそのまま転用できる形にする。
API を決めた後、ドキュメント更新のコストがほぼゼロになるのがこのアプローチの利点。

***

## API Design Checklist

新しい API を設計・変更するたびに、以下を確認する。

### 責務と配置

- [ ] その API はどの層（`core` / `nn` / `optim` / `io`）に属するか明確か？
- [ ] レイヤリングルールに違反しないか？（`mbtorch-architecture` スキルも参照）

### 一貫性

- [ ] 既存の類似 API と命名スタイルが揃っているか？
- [ ] 引数の順序・型・デフォルト値が既存 API と一貫しているか？
- [ ] 戻り値の型・エラーの返し方が既存 API と揃っているか？

### 使いやすさ

- [ ] 必須引数とオプション引数が適切に分かれているか？
- [ ] デフォルト値は「最も一般的なユースケース」に合っているか？
- [ ] 引数が 4 つ以上になる場合は、設定 struct（Config 型など）にまとめることを検討したか？

### エラーと境界

- [ ] 無効な入力（shape 不一致・dtype 非対応など）は早期にエラーになるか？
- [ ] エラーメッセージは原因と対処法が分かる文章になっているか？

### 拡張性

- [ ] 将来の引数追加に耐えられる形か？（Config struct パターンが有効）
- [ ] 変更が破壊的（breaking change）になる場合、その影響範囲を把握しているか？

***

## Error Handling & Messages

### 基本方針

- **入力の検証は早く行う（fail fast）**: shape・dtype・範囲の不一致は、処理の奥深くまで進む前にエラーにする
- **エラーメッセージは「診断情報」として設計する**: 利用者が原因を特定し、修正できるメッセージにする

### 良いエラーメッセージの構造

```
[何が原因か] + [実際の値] + [期待される値・対処法]
```

**例:**
```
// 悪い例
"shape mismatch"

// 良い例
"Linear::forward: input shape [32, 128] does not match weight shape [64, 256]. \
 Expected input last dimension to be 256, got 128."
```

### io/ でのエラー

ファイル I/O（ONNX / safetensors / .mbt）では以下の情報をメッセージに含める:
- ファイルパス
- フォーマット識別子（マジックバイトやバージョン番号）
- サポート対象バージョンと実際のバージョン

**例:**
```
"OnnxLoader: unsupported opset version 18 in 'resnet18.onnx'. \
 Supported versions: 11, 13, 17."
```

***

## Backward Compatibility

既存 API の挙動を変える前に、必ず以下を検討する。

### 優先順位（上から順に検討）

1. **後方互換な拡張** — 引数を追加する場合はオプション引数（デフォルト値あり）として追加し、既存の呼び出しを壊さない
2. **非推奨サイクル** — 挙動の変更や削除が避けられない場合は、非推奨期間（deprecation）を設け、新旧両方を一時的に動かす
3. **破壊的変更** — 最終手段。ユーザーへの影響を事前に説明し、移行ガイドを用意してから行う

### 破壊的変更を提案するとき

1. 影響範囲（どの API が、どのコードに影響するか）を整理してユーザーに伝える
2. 移行手順を具体的なコード例で示す
3. CHANGELOG 相当のドキュメントへの記載を提案する
4. ユーザーの確認を取ってから実装に進む

**ユーザーに黙って挙動を変えることは絶対にしない。** 挙動変更は必ず事前に説明する。

***

## When to Use This Skill

以下のタスクでは必ずこのスキルを有効にすること。

- 新しい公開 API（関数・メソッド・型・トレイト）を追加するとき
- `core/`, `nn/`, `optim/`, `io/` の公開インターフェイスを変更・整理するとき
- `examples/` を書きながら API 形状を決めるとき
- 既存 API の命名・シグネチャがバラバラで、統一したいとき
- 既存 API を変更・削除しようとしているとき（後方互換性の確認）
- 「このメソッドの名前、これで合ってる？」「引数の順番どうしよう」といった API の相談をするとき

***

## Do / Don't

### Do

- API を決める前に「使う側のコード例」を書き、それをシグネチャの根拠にする
- 既存 API の命名・引数スタイル・戻り値パターンと揃える
- 引数が増えそうな API は Config struct でまとめる設計を検討する
- エラーメッセージに「原因・実際の値・期待値」を含める
- 破壊的変更の前に影響範囲と移行パスをユーザーに示す
- API 変更後は tests, README, examples との整合性も確認する

### Don't

- 実装の都合で API の形を決める（利用者視点を忘れる）
- 単発の思いつきで似た機能の API を増やし続ける（重複・断片化させない）
- 似た機能なのに別の命名スタイル・別の引数順を使う（一貫性を壊さない）
- 既存 API の挙動を黙って変える（必ず事前説明と移行案を出す）
- テストと examples を更新せずに API だけ変える
