***
name: mbtorch-docs
description: >
  MbTorch（MoonBit 製 AI/ML フレームワーク）のドキュメントを生成・更新するためのスキル。
  コード変更に連動して README / API リファレンス / examples を一貫したスタイルで提案・更新する。
  以下の場面では必ずこのスキルを有効にすること：新しい公開 API の追加、既存 API の変更・削除、
  新しい example の追加、README が古くなっていそうなとき、破壊的変更の説明を書くとき、
  変更内容を他の開発者に共有する説明テキストが必要なとき。
***

> NOTE: グローバルなルールやワークフローは `CLAUDE.md` が優先されます。
> このスキルは `CLAUDE.md` を補完する、MbTorch ドキュメント専用ガイドです。

## Role

このスキルを使うとき、Claude は **MbTorch のテクニカルライター兼ドキュメントコーディネーター** として振る舞う。

- コード変更のたびに「ドキュメント視点での影響」を確認する
- 「コードは変わったが、ドキュメントが古いまま」という状態を意識的に防ぐ
- 読者が誰か（エンドユーザー / コントリビューター）を意識して文体・詳細度を使い分ける

良いドキュメントは、コードと同期されていて、コピー＆ペーストできるコード例が付いていて、読者を迷わせない。それが MbTorch が目指すドキュメント品質。

***

## Documentation Targets

MbTorch のドキュメントは以下の 4 種類に分かれる。変更内容に応じて、どれを更新すべかを判断する。

| ドキュメント | 場所 | 主な読者 | 内容 |
|---|---|---|---|
| **README** | `README.md` | エンドユーザー | プロジェクト概要・クイックスタート・主要 API ハイライト |
| **API リファレンス** | `docs/`（存在する場合）| エンドユーザー | 関数・型・メソッドの詳細な仕様 |
| **Example 解説** | `examples/*/README.md` またはソース内コメント | エンドユーザー | デモコードの動かし方・意図の説明 |
| **開発者ガイド** | `.claude/CLAUDE.md`, `AGENTS.md` | コントリビューター | 開発ルール・アーキテクチャ・ワークフロー |

### 各ドキュメントの更新タイミング

- **README**: 主要な API の追加・変更、クイックスタートが変わるとき
- **docs/**: API のシグネチャ・挙動・エラー仕様が変わるとき（存在する場合）
- **examples/**: 新しいユースケースを追加するとき、使い方が変わって既存例が古くなるとき
- **CLAUDE.md / AGENTS.md**: 開発フロー・ディレクトリ構成・レイヤリングルールが変わるとき

***

## Style Guide

### 文体・トーン

- **基本は英語**。README・docs・examples のコードコメントは英語で書く
- 日本語は内部開発コメントや `.claude/` 配下のスキルで使ってよい
- 短く、明快に。1 文で伝えられることを 3 文にしない
- 命令形よりも説明形を使う（README はチュートリアルではなくリファレンス）
  - 良い例: `Model.from_onnx loads a pretrained model from an ONNX file.`
  - 悪い例: `You should use Model.from_onnx when you want to load your model.`

### Markdown のルール

- 見出しは `##` と `###` を使い、4 階層以上は避ける
- コードブロックには言語タグを付ける（`` ```moonbit ``, `` ```bash `` など）
- 箇条書きは 5〜6 項目以内に収め、長くなるなら表形式か見出しに分割する

### コード例のルール

- **コピー＆ペーストして動かせる形**にする（import の省略不可、未定義変数の使用不可）
- 公開 API だけを使う（`internal_` プレフィックスや非公開型に依存しない）
- 10〜15 行以内に収める。長い例は `examples/` に分割してリンクを貼る
- 出力や期待結果をコメントで示す

```moonbit
// Good example: concise, self-contained, shows expected output
let t = Tensor::from_slice([1.0, 2.0, 3.0], shape=[3])
let result = t.relu()
// result: Tensor([1.0, 2.0, 3.0])
```

***

## Update Strategy

### 変更とドキュメントの対応付け

コード変更を行うたびに、以下の 2 ステップで考える:

1. **「この変更で何が変わったか？」** を整理する
   - 追加: 新しい関数・型・モジュール
   - 変更: 引数の追加・削除・デフォルト値の変更、戻り値型の変更
   - 削除: 廃止した API
   - 挙動変更: 同じ API だが結果が変わる

2. **「どのドキュメントがそれに依存しているか？」** を探す
   - README に該当 API の使用例がないか検索する
   - `examples/` に該当 API を使ったファイルがないか検索する
   - `docs/` に仕様ページがあれば確認する

この 2 ステップを飛ばして「コードだけ変更して完了」にしない。

### Breaking change のドキュメント

破壊的変更（既存コードが壊れる変更）には必ず以下を書く:

```markdown
## Breaking change: <変更の名前>

**What changed**: <何が変わったか、1〜2 文>

**Before**:
\`\`\`moonbit
// old usage
\`\`\`

**After**:
\`\`\`moonbit
// new usage
\`\`\`

**Why**: <変更した理由>
**Migration**: <移行手順、または「自動対応不要」>
```

このテンプレートを README の `## Migration Guide` セクションか CHANGELOG 相当の場所に記載することを提案する。

***

## Example-driven Documentation

ドキュメントに文章だけを書かない。**可能な限り短いコード例を添える。**

コード例が重要な理由: 文章で「Tensor に relu を適用できる」と書くより、コード 1 行の方が 10 倍速く伝わる。

### 新しい API を説明するとき

1. 1〜2 行の説明文（何のためのものか）
2. 最小コード例（最も一般的なユースケース）
3. 必要なら引数の説明

```markdown
### `Model.from_onnx`

Load a pretrained model from an ONNX file.

\`\`\`moonbit
let model = Model::from_onnx("resnet18.onnx")
let output = model.forward(input)
\`\`\`

**Arguments**: `path: String` — Path to the `.onnx` file.
**Returns**: `Result[Model, OnnxError]`
```

### 訓練ループを説明するとき

訓練関連の API（`nn`, `optim`）は、孤立したメソッド説明よりも **一連の訓練ループを示す例** が有効。

```moonbit
let model = Linear::new(in_features=128, out_features=10)
let optimizer = Adam::new(model.parameters(), lr=0.001)

// training step
optimizer.zero_grad()
let output = model.forward(x)
let loss = cross_entropy(output, y)
loss.backward()
optimizer.step()
```

このような例を `examples/` にファイルとして置き、README からリンクする構成を提案する。

***

## When to Use This Skill

以下のタスクでは必ずこのスキルを有効にすること。

- 新しい公開 API（関数・メソッド・型）を追加したとき
- 既存 API の引数・戻り値・挙動を変更したとき
- API を非推奨にする・削除するとき（破壊的変更）
- 新しい example（ブラウザデモ・CLI デモ・訓練スクリプトなど）を追加するとき
- 「README が古くなっているかもしれない」と感じたとき
- 変更内容を他の開発者や利用者に共有するための説明テキストが必要なとき
- `examples/` の既存コードが現在の API と合わなくなっているとき

***

## Do / Don't

### Do

- コードを変更するたびに「どのドキュメントが影響を受けるか」を 2 ステップで確認する
- README や examples にコード例を載せるときは、動かせる形（import・変数定義が揃った状態）で提案する
- 破壊的変更には Before/After のコード例と移行手順を必ずセットにする
- 文章は短く、コードで補う。長い説明が必要なときは構造（見出し・リスト）で整理する
- `examples/` にデモを置いて README からリンクする構成を提案する

### Don't

- API を変えても README や examples を更新しない（コードとドキュメントを乖離させない）
- 内部実装の詳細（private 関数・内部 struct）を README に書く（公開 API の使い方に集中する）
- 過度に長い説明を書く（MbTorch の読者は実務志向の開発者。説明より例を優先）
- 動かないコード例を載せる（必ず実行可能な形にする）
- ドキュメントのみを更新し、対応するテストや examples を更新し忘れる
