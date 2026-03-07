***
name: moonbit-investigating-errors
description: >
  Investigate MoonBit compiler errors and error codes, explain their causes
  in plain language, and propose concrete resolution plans. Use this skill
  whenever MoonBit compiler errors, error codes (E0xxx/E3xxx/E4xxx), build
  failures, or diagnostics need to be understood and resolved.
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
- Get a step-by-step plan to resolve the error, including code-level changes.
- Prioritize which errors to fix first when there are multiple issues.

It complements, but does not replace, the separate skill that reviews code
against the language specification. This skill is focused on **errors** and
**troubleshooting**.

## Resources

When using this skill, rely on the resource files shipped with the skill
instead of guessing. Read the relevant resource before forming a hypothesis.

### `resources/index.md` — Error Code Index

A categorized index of all ~235 MoonBit error codes, organized by range and
theme for quick lookup.

| Section | Contents |
|---------|----------|
| E0xxx Warnings | Unused items, code quality, deprecated syntax, FFI/WASM warnings |
| E1xxx Internal | Internal compiler errors (ICE) |
| E3xxx Syntax | Lexing, parsing, signature, and syntax errors |
| E4xxx Type/Semantic | Type mismatches, visibility, traits, methods, functions, modules |

Each entry includes the warning/error name and a one-line description.
Use this as the starting point to locate the right `error_codes/EXXXX.md` file.

Source: derived from `docs/moonbit-docs/error_codes/` (official MoonBit docs).

### `resources/error_codes/*.md` — Per-Error-Code Details

One file per error code (e.g. `E0001.md`, `E4014.md`), containing:
- The official description of the error.
- Typical causes and explanations.
- Minimal code examples that reproduce the error.
- Recommended fixes.

There are ~235 files covering the full range of MoonBit compiler diagnostics.

Source: identical copies from `docs/moonbit-docs/error_codes/` (official MoonBit docs).

### `resources/troubleshooting.md` — Diagnosis Strategies

Practical strategies for investigating errors, organized as:

| Section | What it covers |
|---------|---------------|
| Quick Triage | Decision tree: error code present? runtime error? build system error? |
| Priority Rules | Fix syntax errors first, fix earliest error first, fix root causes first |
| Error Categories | First actions for E0xxx warnings, E3xxx syntax, E4xxx type/semantic |
| Cascading Error Strategy | 4 patterns: missing import, type definition, error handling, visibility |
| Common Patterns | "Code used to work", "copied from tutorial", "types look the same" |
| Error Handling Reference | `type!`, `raise`, `try`/`catch`, `try?`, `!!`, common mistakes table |
| Build & Toolchain | `moon.pkg.json` problems, test file conventions (`_test.mbt` vs `_wbtest.mbt`) |

Source: compiled from `docs/moonbit-docs/` official language documentation and error code descriptions.

## How to handle an error report

When the user provides error messages, logs, or screenshots:

### 1. Quick triage

Read `resources/troubleshooting.md` (Quick Triage section) and classify:
- **Error code present** → look up in `resources/index.md` → read `resources/error_codes/EXXXX.md`
- **Runtime error** → check for panic, abort, uncaught raise; look at stack trace
- **Build system error** → check `moon.pkg.json`, `moon.mod.json`

### 2. Extract and look up error codes

- Identify all error codes (e.g. `E0001`, `E4014`).
- Capture file names, line/column numbers, and full error text.
- Read the corresponding `resources/error_codes/EXXXX.md` for each code.
- Note any relationships between errors (one causing another).

### 3. Check for cascading errors

Read `resources/troubleshooting.md` (Cascading Error Strategy section).
Common cascading patterns:
- **Missing import** (E4020) → type mismatches (E4014) → method not found (E4015)
- **Wrong type definition** → multiple E4014/E4015/E4018 errors
- **Missing `raise` in signature** → unused try (E0023) → return type mismatch (E4014)
- **Visibility too restrictive** → E4001 → construction/pattern matching fails externally

When cascading is detected, identify and fix the **root cause** first.

### 4. Form hypotheses

- Propose 1-3 plausible causes for each error based on the code and error docs.
- Distinguish between "very likely" and "less likely" causes.
- Be explicit when uncertain.

### 5. Propose a resolution plan

For each error (or group of related errors):
- Explain in plain language what the error means.
- Describe why it is happening in the user's specific context.
- Provide a **step-by-step plan** with concrete code changes.
- Show before/after code snippets when helpful.

### 6. Prioritize fixes

When multiple errors are present:
1. Fix **syntax errors (E3xxx)** first — they prevent parsing and cause cascading type errors.
2. Fix the **earliest error** in the file — later errors often depend on it.
3. Fix **root-cause errors** — if error A causes error B, fix A first.

Suggest an order so the user can rebuild after each key fix.

## How to use other skills together

- If the error stems from code that doesn't follow MoonBit idioms, recommend
  the "MoonBit code evaluation" skill for a thorough review.
- Always provide immediate, concrete fixes for the specific error at hand —
  do not delegate all work to other skills.

## What NOT to do

- Do NOT invent error codes or meanings not documented in the resources.
- Do NOT suggest MoonBit features or flags that do not exist.
- Do NOT only restate the error message; always add explanation and a
  concrete resolution plan.
- Do NOT change program behavior in major ways unless the user explicitly
  allows it; explain why if a behavior change is required.
