# DEVELOPER FLOW （開発フロー）

MbTorch では、**1 モジュールごとに同じ開発サイクル**を回すことを基本とします。  
ここでは、例として `core/tensor` に基本テンソル演算（`add`, `mul`, `matmul`）を追加する場合のフローを示します。

***

## 全体像（1 モジュール分のサイクル）

1. アーキテクチャ確認  
   → `mbtorch-architecture` ＋ `CLAUDE.md`  

2. テスト設計（TDD）  
   → `mbtorch-tdd`  

3. API 形状の確認（必要なら）  
   → `mbtorch-api-design`  

4. 実装（必要に応じて言語仕様・エラー調査）  
   → 通常実装 ＋ `moonbit-evaluating-code` / `moonbit-investigating-errors`  

5. ドキュメント・examples・整理  
   → `mbtorch-docs` ＋ `mbtorch-examples` ＋（必要に応じて）`mbtorch-refactor`  

***

## 1. どこに置くか決める（アーキテクチャ）

まず、「どのディレクトリ・ファイルに実装を置くか」を決めます。

```text
/plan
スキル: mbtorch-architecture, mbtorch-tdd
内容:
core/tensor に「基本テンソル演算（add, mul, matmul）」を実装したいです。
どのディレクトリ・ファイルに何を追加すべきか、
既存構造に沿って簡単にプランを作ってください。
```

このステップで:

- `core/tensor/types.mbt`, `core/tensor/ops_basic.mbt` などのファイル案を決める  
- `CLAUDE.md` に書かれたレイヤリングルールを再確認する  

***

## 2. 先にテストを書く（mbtorch-tdd）

次に、**実装より先にテストだけ**を書きます（TDD）。

```text
/plan
スキル: mbtorch-tdd, mbtorch-architecture
内容:
core/tensor/ops_basic.mbt に add, mul, matmul を実装する前提で、
tests/ 配下に置くべきテストファイル名とテストケース案を出してから、
実際のテストコードを書いてください。
まだ本体実装は書かないでください。
```

ここで:

- テストファイル名（例: `tests/core_tensor_ops.mbt`）を決める  
- 「shape と値の検証」「matmul の shape 」「エラーケース」などのテストを用意する  

***

## 3. API デザインを軽くチェック（必要なとき）

新しい公開 API の形（コンストラクタやメソッド名など）を固めたいときは、API 設計スキルを併用します。

```text
/plan
スキル: mbtorch-api-design, mbtorch-tdd
内容:
Tensor の基本コンストラクタと add/mul/matmul の API を設計したいです。
まずはユーザー視点のサンプルコードを書いてから、
最終的な関数シグネチャ案を出してください。
その後、さきほどのテストコードを API に合わせて調整してください。
```

ポイント:

- 「サンプルコード → シグネチャ → テスト」の順でそろえる  
- 後で README や `examples/` にも流用しやすくなる  

***

## 4. 実装を書く（＋ MoonBit 言語チェック・エラー調査）

テストが赤の状態になったら、テストを通すための最小実装を書かせます。

```text
/plan
スキル: mbtorch-tdd, mbtorch-architecture
内容:
すでに作成した tests/core_tensor_ops.mbt が赤の状態です。
このテストを通すために必要な最小限の実装を
core/tensor/types.mbt と core/tensor/ops_basic.mbt に追加してください。
実装後に必要なリファクタ案があれば separate step として提案に留めてください。
```

### MoonBit 言語仕様の確認が必要なとき

実装が MoonBit として正しいか不安な場合は、言語評価スキルを使います。

```text
/moonbit-evaluating-code
この core/tensor/ops_basic.mbt の実装が MoonBit の言語仕様的に正しいか、
必要なら修正コードを提案してください。
```

### コンパイルエラーが出たとき

コンパイルエラーの意味や対処を知りたい場合は、エラー調査スキルを使います。

```text
/moonbit-investigating-errors
MoonBit のビルドで以下のエラーが出ます。
エラーコードの意味と具体的な修正手順を教えてください。

<エラーログ貼り付け>
```

***

## 5. ドキュメント・examples・リファクタ

機能が動くようになったら、「どう見せるか」と「どう整理するか」を仕上げます。

### 5-1. README / docs を更新する（mbtorch-docs）

```text
/plan
スキル: mbtorch-docs
内容:
Tensor の基本演算（add, mul, matmul）とコンストラクタを追加しました。
README のどこを更新すべきか提案し、
簡単なコード例付きで更新案を出してください。
```

### 5-2. example を追加する（mbtorch-examples）

```text
/plan
スキル: mbtorch-examples
内容:
core/tensor の基本機能を試せる最小の CLI example を
examples/basic_tensor_ops/ として追加したいです。
ディレクトリ構成と main.mbt, README.md の内容案を書いてください。
```

### 5-3. コードが膨らんできたら整理する（mbtorch-refactor）

しばらく開発を進めて `core/tensor` 周辺が肥大化してきたら、リファクタリングスキルを使います。

```text
/plan
スキル: mbtorch-refactor, mbtorch-architecture, mbtorch-tdd
内容:
core/tensor 関連のコードが肥大化してきたので、
mbtorch-refactor のルールに従って問題点の整理と
安全な分割プラン（ファイル構成と手順）を出してください。
その後、最初の小さなリファクタステップだけコード変更案を出してください。
```

***

## まとめ：1 モジュール開発の「型」

Tensor の基本演算モジュールを例にした場合、典型的なフローは次の通りです。

1. `/plan + mbtorch-architecture`  
   → 実装場所とファイル名を決める  

2. `/plan + mbtorch-tdd`  
   → テストファイル＋テストコードを書く（実装はまだ）  

3. （必要なら）`/plan + mbtorch-api-design`  
   → API 形状とサンプルコードを固める  

4. `/plan + mbtorch-tdd`（＋ `moonbit-evaluating-code` / `moonbit-investigating-errors`）  
   → 実装してテストを通す  

5. `/plan + mbtorch-docs` ＋ `/plan + mbtorch-examples`  
   → README / examples を更新する  

6. 規模が大きくなってきたら `/plan + mbtorch-refactor`  
   → 構造の整理・分割を行う  

この「型」を `core/tensor` で一度回してみて、慣れてきたら `core/autograd`, `nn`, `optim`, `io` でも同じパターンを適用していきます。
