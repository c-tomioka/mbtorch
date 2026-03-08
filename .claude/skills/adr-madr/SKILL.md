***
name: adr-madr
description: >
  MADR (Markdown Architectural Decision Records) 形式で ADR を作成・更新するスキル。
  設計上の重要な決定を一貫したフォーマットで記録する。
  次のような場面では必ずこのスキルを使うこと：ADR の新規作成、既存 ADR の更新、
  git log や PR/Issue からの ADR 候補抽出、アーキテクチャ決定の文書化、
  「ADR」「設計決定」「意思決定記録」「decision record」といったキーワードが含まれるリクエスト。
***

> NOTE: グローバルなルールやワークフローは `CLAUDE.md` が優先されます。
> このスキルは設計決定の記録に特化した ADR 作成ガイドです。

## Role

このスキルを使うとき、Claude は **ADR フォーマットの番人** として振る舞う。

- 設計判断そのものを勝手に上書きせず、ユーザーが説明した内容を構造化・整理して MADR にマッピングする
- 事実が不明な場合は、勝手に埋めずに「(assumption)」「(情報不足)」と明記する
- 1 回のスキル呼び出しでは 1 個の ADR だけを完成させる

---

## 利用パターン

### パターン 1: 新規 ADR の作成

ユーザーが決定の背景・検討した案・最終案を自由形式で説明する。
スキルは MADR フォーマットに沿った Markdown を生成する。

### パターン 2: git log / PR / Issue からの ADR 逆算

ユーザーが関連コミットや PR/Issue のテキストを渡す。
スキルは推測を含めて Context / Considered Options / Decision Outcome を構成し、
推測箇所には `(assumption based on git history)` という注記を付ける。

### パターン 3: 既存 ADR の更新

ユーザーが既存 ADR の Markdown と変更要件を渡す。
スキルは更新後の完全な ADR を再生成する。

---

## ADR フォーマット (MADR 準拠・ミニマル版)

以下のセクション構成で Markdown を生成する。テキスト量は A4 1〜2 ページ程度を目安にし、冗長にしない。

### 1. タイトル行

```
# {ID}: {短いタイトル}
```

- ユーザーが ID を指定した場合はそれを使う
- 未指定の場合は `XXXX` をプレースホルダとして使い、後から置き換えてもらう

**例:** `# 0003: Use JSON-based .mbt format for initial model IO`

### 2. メタデータ

タイトル直下に以下を bullet で記述する:

```
* Status: {proposed | accepted | rejected | deprecated | superseded}
* Date: {YYYY-MM-DD}
* Deciders: {名前やロールのカンマ区切り}
* Consulted: {任意}
* Informed: {任意}
```

**デフォルト値** (指定がない場合):
- Status: `proposed`
- Date: 現在日付
- Deciders: `@c-tomioka, Claude Code`
- Consulted / Informed: 省略可

### 3. Context and Problem Statement

```
## Context and Problem Statement
```

- 2〜5 文程度で背景・問題・前提条件・制約を書く
- 関連 Issue / PR / ドキュメントがあれば bullet で並べる

### 4. Decision Drivers

```
## Decision Drivers
```

- 箇条書きで 3〜7 個程度
- 性能・開発コスト・保守性・DX・MoonBit/WASM 前提など、意思決定に効いている要因

### 5. Considered Options

```
## Considered Options
```

- 各選択肢を bullet で 1 行ずつ列挙する

**例:**
```
- Use JSON-based .mbt format first, introduce binary later
- Start with binary .mbt format
- Reuse ONNX / safetensors only, no custom format
```

### 6. Decision Outcome

```
## Decision Outcome
```

- 最初の行: `Chosen option: "...", because` の形式
- その後に bullet で 3〜7 個の理由

### 7. Pros and Cons of the Options

```
## Pros and Cons of the Options
```

各オプションごとにサブ見出しを付ける:

```
### Option 1: {タイトル}

#### Pros
- ...

#### Cons
- ...
```

情報が曖昧な場合は推測して良いが `(assumption)` を付ける。

### 8. Links

```
## Links
```

- 関連する Issue, PR, README, ソースファイルパスなどを bullet で列挙
- 既存 ADR との関係がある場合: `- Supersedes: 0002: ...` や `- Related ADRs: 0001, 0004`

---

## ファイル命名規則

- ディレクトリ: `docs/adr/`
- ファイル名: `{4桁ゼロ埋め番号}-{kebab-case-title}.md`
- 例: `0003-use-json-based-mbt-format.md`
- ID 未指定の場合: `XXXX-{kebab-case-title}.md`

---

## ワークフロー

1. **入力を確認する**: ユーザーの入力（自由テキスト、git log、PR/Issue テキスト、既存 ADR）を読み取る
2. **既存 ADR を確認する**: `docs/adr/` ディレクトリに既存 ADR があれば番号の重複を避ける
3. **ADR を構成する**: 上記フォーマットに沿って各セクションを埋める
4. **情報不足を明示する**: 不明な箇所は `(assumption)` や `(要確認)` を付ける
5. **ファイルパス案を提示する**: 命名規則に従ったパスを提案する
6. **ユーザーに確認する**: 内容とファイルパスについて確認を取ってからファイルを書き出す

---

## 注意事項

- 日付は `YYYY-MM-DD` 形式で統一する
- 1 ファイル 1 決定を厳守する
- 「既存の ADR を前提にしている決定」の場合は Context または Links で参照を明記する
- MbTorch リポジトリの場合、CLAUDE.md のレイヤリングルールやアーキテクチャ方針と整合する ADR を書く
