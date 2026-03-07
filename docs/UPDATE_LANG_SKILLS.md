#  UPDATE_LANG_SKILL（言語専用スキル更新方法）

本プロジェクトにある MoonBit 言語仕様に関するスキルの更新方法

1. `docs/moonbit-docs` ディレクトリに、公式ドキュメントの next ディレクトリ配下から必要なリソースだけをコピー
2. 各言語専用スキルの SKILL.md ファイルを指定して、skill-creator を使って更新する

## moonbit-evaluating-code スキル更新フロー

`docs/moonbit-docs` に以下を配置

- language (内包する error_codes は抜き出しておく)
- example

`.claude/skills/moonbit-evaluating-code/SKILL.md` を指定して、Claude Code で `/skill-creator` を使用して更新する。

Prompt:
```
/skill-creator

moonbit-evaluating-code スキルを更新してください。以下の公式言語リファレンスを参照し、必要があればコピーし、本スキルに最適化した状態で `.claude/skills/moonbit-evaluating-code/resources` にファイルを作成・更新してください。
必要に応じて、SKILL.md ファイル内容も最適化してください。
```

## moonbit-investigating-errors スキル更新フロー

`docs/moonbit-docs` に以下を配置

- error_codes

`.claude/skills/moonbit-investigating-errors/SKILL.md` を指定して、Claude Code で `/skill-creator` を使用して更新する。

Prompt:
```
/skill-creator

moonbit-investigating-errors スキルを更新してください。以下の公式言語リファレンスを参照し、必要があればコピーし、本スキルに最適化した状態で `.claude/skills/moonbit-investigating-errors/resources` にファイルを作成・更新してください。
必要に応じて、SKILL.md ファイル内容も最適化してください。
```
