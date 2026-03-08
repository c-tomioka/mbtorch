# 0003: TDD-first 開発ワークフローと Claude Code スキル体系の採用

* Status: accepted
* Date: 2026-03-07
* Deciders: @c-tomioka, Claude Code

## Context and Problem Statement

MbTorch の開発では、Claude Code を主要な開発パートナーとして活用する。AI アシスタントが一貫した品質でコードを生成・修正するためには、開発ワークフローを標準化し、AI が従うべきルールを明文化する必要があった。

特に、テストなしで実装が先行すると回帰バグや仕様のずれが発生しやすいため、TDD を開発の基本原則として定めるかどうかが論点となった。

## Decision Drivers

- AI アシスタントとの協働で一貫した開発品質を保ちたい
- 回帰バグを早期に検出したい
- 新しいモジュール追加時に「何から始めるか」を標準化したい
- Claude Code のスキル機能を活用して、タスクごとの専門知識を構造化したい
- 開発フローをドキュメントとして残し、再現可能にしたい

## Considered Options

- TDD-first ワークフロー ＋ タスク別 Claude Code スキル体系
- 実装先行 ＋ 後付けテスト
- テストなし（動作確認のみ）

## Decision Outcome

Chosen option: "TDD-first ワークフロー ＋ タスク別 Claude Code スキル体系", because:

- テストを先に書くことで、実装の仕様が明確になる
- Claude Code がスキルに従って動作することで、一貫した開発パターンが保たれる
- 9 つのスキル（architecture, tdd, api-design, docs, examples, refactor, io-formats, evaluating-code, investigating-errors）で開発ライフサイクルをカバーできる
- `docs/DEVELOPER_FLOW.md` でフローを明文化し、誰でも（AI 含む）同じ手順で開発できる

## Pros and Cons of the Options

### Option 1: TDD-first ＋ Claude Code スキル体系

#### Pros
- テストが仕様のドキュメントとして機能する
- AI が生成するコードの品質が安定する
- モジュールごとに同じサイクルを回せる再現性
- スキルにより、タスクの種類に応じた専門的なガイダンスが得られる

#### Cons
- テストを先に書く初期コストが発生する
- スキルの作成・メンテナンスにコストがかかる
- 小さな変更でもフロー全体を意識する必要がある

### Option 2: 実装先行 ＋ 後付けテスト

#### Pros
- 初期の開発速度が速い
- プロトタイピングに向いている

#### Cons
- テストが後回しになり、書かれない場合がある (assumption)
- 仕様が曖昧なまま実装が進むリスク
- 回帰バグの検出が遅れる

### Option 3: テストなし

#### Pros
- 最も初期コストが低い

#### Cons
- 品質の保証がない
- リファクタリング時に安全性が担保されない
- ML フレームワークでは数値計算の正確性が重要であり、テストなしは致命的

## Links

- Commit: `5e00790` 各種スキル作成、および、開発フローメモ追加
- Commit: `6f70d76` 開発フローメモ更新
- Code: `docs/DEVELOPER_FLOW.md`
- Code: `.claude/skills/` (9 スキル)
- Code: `.claude/CLAUDE.md` (Coding and design principles セクション)
- Related ADRs: 0001, 0002
