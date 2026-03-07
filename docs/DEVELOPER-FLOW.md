# 開発フローメモ

このドキュメントは、mbtorch プロジェクトで **AI 中心 (Claude Code Plan モード ＋ Agent Skills)** による開発を行うための標準フローをまとめたものです。  
tensor / autograd / nn などの各モジュールを、ほぼ同じパターンで実装・テストしていくことを想定しています。

***

## 全体像：1 モジュール開発の標準フロー

1. Plan モードでモジュールのゴールと作業ステップを決める  
2. mbtorch 系スキルで API・内部設計・TDD 方針を固める  
3. 実装・テスト・サンプル・ドキュメントを段階的に整える  
4. moonbit-* スキルで MoonBit 言語仕様／エラー観点から仕上げる

この 1〜4 を、tensor / autograd / nn など個々のモジュールに対して繰り返す。

***

## 1. Plan モードでモジュールごとの計画を作る

### 1-1. Plan モードの目的

- `.claude/CLAUDE.md` と `.claude/skills/` の SKILL.md を読ませた上で、  
  - どのファイルを触るか  
  - どのスキルを使うか  
  - どんな順番で実装・テスト・ドキュメント化するか  
  を **plan.md** のような形で整理させる。

### 1-2. 典型的な指示例（tensor モジュール）

CLI から Plan モードでプロジェクトルートに入り、次のように指示するイメージ:

```text
このリポジトリで tensor モジュールの最初のバージョン（MVP）を実装したい。

前提:
- .claude/CLAUDE.md と .claude/skills/ 配下の SKILL.md を全て読んで、
  mbtorch プロジェクト全体の設計方針と各スキルの役割を理解してください。
- 既存の mbtorch.mbt, mbtorch_test.mbt, mbtorch_wbtest.mbt,
  cmd/main/main.mbt なども読み、tensor がどこに位置づくかを把握してください。

やってほしいこと:
- 以下を含む plan.md（または同等の計画）を作成してください:
  - tensor モジュール MVP のゴールの定義
  - 影響するファイル一覧
  - 実装ステップ（API設計 → 型/内部設計 → 実装 → テスト → ドキュメント）
  - 各ステップで利用するスキル
    (例: /mbtorch-api-design, /mbtorch-architecture, /mbtorch-tdd,
         /mbtorch-docs, /moonbit-evaluating-code, /moonbit-investigating-errors)
- まだコードは編集せず、「読む」「考える」「計画を書く」に専念してください。
```

Plan モードでは、まず **仕様理解と計画作成** に集中させる。  
コード編集は、Plan でステップが固まってから行う。

***

## 2. mbtorch 系スキルで設計と TDD を固める

Plan ができたら、そのステップに沿って mbtorch-* スキルを呼んでいく。

### 2-1. API 設計（/mbtorch-api-design）

目的: tensor / autograd / nn などの **パブリック API** を先に決める。

指示例:

```text
/mbtorch-api-design

tensor モジュールの MVP API を設計してください。
前提:
- スカラー/1D/2D の基本演算（加算・乗算・ブロードキャスト）だけに絞ります。
- MoonBit で自然な API として設計してください。
- 既存の mbtorch.mbt と整合性が取れるようにしてください。

出力してほしいもの:
- 代表的なユースケースとサンプル呼び出しコード
- パブリック API 関数一覧（名前・シグネチャ・簡単な説明）
- 必要な型（struct や trait など）の一覧
```

### 2-2. 内部アーキテクチャ設計（/mbtorch-architecture）

目的: API を支える **内部構造・ファイル構成** を決める。

指示例:

```text
/mbtorch-architecture

さきほど決めた tensor の API を実装するための内部構造を設計してください。

要件:
- どのファイルに何を書くか（mbtorch.mbt と分離するか、tensor.mbt を作るかなど）
- struct / trait / helper 関数の役割分担
- 他の mbtorch モジュールとの依存関係の制約（循環を避ける方針など）
```

### 2-3. TDD 方針の決定（/mbtorch-tdd）

目的: **テストケースとテストファイル構成** を先に決める。

指示例:

```text
/mbtorch-tdd

tensor モジュールの MVP API を TDD で実装するためのテスト方針を作ってください。

前提:
- 既存の mbtorch_test.mbt / mbtorch_wbtest.mbt を参照し、
  blackbox / whitebox テストの使い分けを踏襲してください。

出力してほしいもの:
- テストケース一覧（入力 / 期待される出力）
- どのテストをどのファイルに書くか（blackbox/whitebox）
- 実装との進め方（テスト→実装の順序）に関する簡単な指針
```

***

## 3. 実装・テスト・ドキュメントのサイクル

### 3-1. 実装（必要に応じて /mbtorch-refactor）

目的: 設計済みの API/内部構造/TDD 方針に従って実装を進める。

指示例:

```text
上で決めた tensor の API とアーキテクチャ、TDD 方針に従って、
まずは core 部分（構造体定義と基本の加算・乗算）を実装してください。

要件:
- 既存の mbtorch.mbt / mbtorch_test.mbt / mbtorch_wbtest.mbt と整合する形にしてください。
- 差分を分かりやすく出してください（どのファイルのどの位置に何を追加・変更するか）。

必要に応じて /mbtorch-refactor スキルを使い、
既存コードとの統合・リファクタリングも提案してください。
```

### 3-2. テスト実装（/mbtorch-tdd）

目的: 事前に決めたテストケースを具体的なテストコードに落とす。

指示例:

```text
/mbtorch-tdd

tensor 実装に対して、先ほど計画したテストケースを
mbtorch_test.mbt および mbtorch_wbtest.mbt に追加してください。

要件:
- 各テストごとに、簡単な説明コメントを付けてください。
- 既存のテストスタイルを踏襲しつつ、必要に応じて整理・リファクタも提案してください。
```

### 3-3. サンプルコードとドキュメント（/mbtorch-examples, /mbtorch-docs）

目的: 利用者視点の **example と README** を整える。

サンプルコード（/mbtorch-examples）:

```text
/mbtorch-examples

tensor モジュールの簡単な使用例を 2〜3 個作成してください。
ターゲット:
- README.mbt.md に載せられるレベルのサンプル
- スカラー/1D/2D の基本演算を一通り見せるもの
```

ドキュメント（/mbtorch-docs）:

```text
/mbtorch-docs

tensor モジュールの API と制約を README.mbt.md に追記してください。

要件:
- 初心者にも分かる概要
- 代表的な API とその使い方
- 制約や注意点があれば明示
```

***

## 4. MoonBit 言語仕様・エラー観点での仕上げ

### 4-1. 言語仕様・スタイルチェック（/moonbit-evaluating-code）

目的: 実装済みコードが **MoonBit 言語仕様と公式 example に沿っているか** を確認する。

指示例:

```text
/moonbit-evaluating-code

tensor 関連の MoonBit コード（mbtorch.mbt, tensor 関連のテストファイルなど）を
すべてレビューしてください。

やってほしいこと:
- MoonBit の言語仕様と公式 example に沿っているかを確認
- 文法・型・モジュール構成の観点で問題点があれば列挙
- 必要なら具体的な修正差分（before/after のコード）を提示
- 非推奨または非イディオマティックな書き方があれば、より良い書き方を提案
```

### 4-2. エラー調査・対応計画（/moonbit-investigating-errors）

目的: `moon build` やテスト実行で出る **エラーの原因と対応手順** を整理する。

指示例:

```text
/moonbit-investigating-errors

moon build（またはテスト実行）のログを貼るので、
出ている MoonBit のエラーコードとメッセージについて:

- エラーごとの意味の説明
- 考えられる原因候補（1〜3 個）
- 修正のためのステップバイステップの対応計画

を出してください。
```

典型的な流れ:

1. `moon build` / テスト実行 → エラー発生  
2. ログを `/moonbit-investigating-errors` に渡して原因と対応方針を出してもらう  
3. 対応方針に沿って実装を修正  
4. 再度 `/moonbit-evaluating-code` で仕上げチェック  
5. 必要であれば Plan モードに戻り、「tensor の次の拡張」への計画に反映する

***

## 5. 1 モジュール開発サイクルのまとめ

tensor / autograd / nn など、各モジュール開発の最小サイクルは次の通り:

1. **Plan**  
   - Plan モードで「モジュール MVP のゴール・ステップ・使用スキル」を整理する。

2. **Design**  
   - `/mbtorch-api-design` で API 設計  
   - `/mbtorch-architecture` で内部構造設計  
   - `/mbtorch-tdd` でテスト方針

3. **Implement & Test**  
   - 実装（必要に応じて `/mbtorch-refactor`）  
   - `/mbtorch-tdd` でテストコード実装  
   - `/mbtorch-examples`, `/mbtorch-docs` でサンプルとドキュメント

4. **Polish**  
   - `/moonbit-evaluating-code` で MoonBit 的な正しさとスタイルの最終チェック  
   - `/moonbit-investigating-errors` でエラー原因と対応計画を整理・修正

この 1〜4 を繰り返しつつ、適宜 Plan モードで全体アーキテクチャやロードマップを更新することで、AI 主体の開発フローを安定して回すことを目指す。
