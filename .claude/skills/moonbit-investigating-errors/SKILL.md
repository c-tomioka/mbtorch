***
name: moonbit-investigating-errors
description: >
  Investigate MoonBit compiler errors and error codes, explain their causes
  in plain language, and propose concrete resolution plans.
tags:
  - moonbit
  - errors
  - troubleshooting
  - diagnostics
***

## Purpose

This skill is used to **investigate MoonBit compiler errors and error codes**,
explain what they mean, why they occur, and how to fix them with concrete
actionable steps.

Use this skill when you need to:

- Understand the meaning of a MoonBit error code or compiler error message.
- Diagnose the most likely causes based on code snippets and context.
- Get a step‑by‑step plan to resolve the error, including code‑level changes.
- Prioritize which errors to fix first when there are multiple issues.

It complements, but does not replace, the separate skill that reviews code
against the language specification. This skill is focused on **errors** and
**troubleshooting**.

## What resources to rely on

When using this skill, you MUST rely on the MoonBit error resources shipped
with the skill instead of guessing.

At minimum, the following resource files will be available:

- `resources/index.md`  
  An index of MoonBit error codes and categories, with links to detailed entries.

- `resources/error-codes/*.md`  
  One file per error code (for example `E0001.md`, `E0002.md`), containing:
  - The official description of the error.
  - Typical causes.
  - Minimal examples that reproduce the error.
  - Recommended fixes.

- `resources/troubleshooting.md`  
  General troubleshooting strategies, checklists, and decision trees for
  handling multiple errors, cascading failures, and ambiguous messages.

If you are unsure about an error, look it up in `error-codes/*.md` via
`index.md` before forming a hypothesis.

## How to handle an error report

When the user provides error messages, logs, or screenshots, follow this
process:

1. Extract the key information  
   - Identify all MoonBit error codes present (e.g., `E0001`, `E0123`).
   - Capture the full error message text, the file name, and the line/column
     if available.
   - Note any repeated or cascading errors that may have a common root cause.

2. Look up each error code  
   - Use `resources/index.md` to find the corresponding `error-codes/*.md`
     entry for each code.
   - Read the official description, typical causes, and examples for that code.
   - If multiple codes are present, look for relationships (e.g., one error
     is a consequence of another).

3. Form hypotheses about the causes  
   - Based on the user’s code snippet and the error documentation, propose
     1–3 plausible causes for each error.
   - Make it explicit when you are not fully certain, and clearly distinguish
     between “very likely” and “less likely” causes.

4. Propose a resolution plan  
   For each error (or group of related errors):

   - Explain, in plain language, what the error means.
   - Describe why it is happening in the user’s specific context.
   - Provide a **step‑by‑step plan** to fix it, which may include:
     - Changes to specific lines of code.
     - Adjustments to types, traits, or module structure.
     - Build or toolchain configuration fixes, if relevant.
   - When appropriate, show a small before/after code snippet to illustrate
     the fix.

5. Prioritize the fixes  
   - If there are many errors, indicate which ones should be addressed first
     (typically the ones that are root causes).
   - Suggest an order of operations, so the user can re‑build and re‑run
     after each key fix.

## How to use other skills together

- If you find that the error is caused by code that does not follow the
  language specification or idiomatic patterns, you may recommend using the
  separate “MoonBit code evaluation” skill for a more thorough review.
- However, in this skill you should still provide immediate, concrete fixes
  that resolve the specific error at hand.

Do NOT delegate all work to other skills; always give a direct answer for
the given errors.

## What NOT to do

- Do NOT invent error codes or meanings that are not documented in the
  provided resources.
- Do NOT suggest features or flags that do not exist in MoonBit.
- Do NOT only restate the error message; always add your own explanation
  and a concrete resolution plan.
- Do NOT change the behavior of the program in major ways unless the user
  explicitly allows such changes; if a behavior change is required, explain
  why.

## Example invocations

Use this skill when you receive prompts like:

- “/moonbit-investigating-errors MoonBitのコンパイル時に `E0123` が出ます。このエラーの意味と直し方を教えてください。ログを貼ります。”
- “/moonbit-investigating-errors 複数のエラーコードが出ていて、どれから直せば良いか分かりません。原因と優先度を整理してほしいです。”
- “/moonbit-investigating-errors このコードをビルドすると error_codes にあるエラーが出続けるので、原因を特定して修正手順を提案してほしいです。”
- “/moonbit-investigating-errors MoonBit のエラーメッセージの意味が分からないので、日本語で解説してもらえますか？”

When responding, always:

- Tie your explanation back to the error code documentation.
- Make your reasoning explicit.
- Provide concrete next steps the user can take to resolve the errors.
