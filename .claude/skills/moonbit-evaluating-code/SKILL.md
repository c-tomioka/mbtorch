***
name: moonbit-evaluating-code
description: >
  Evaluate MoonBit code and designs against the official MoonBit language
  specification and examples, then propose compliant and idiomatic fixes.
tags:
  - moonbit
  - language-spec
  - code-review
  - best-practices
***

## Purpose

This skill is used to **evaluate and improve MoonBit code and small designs**
so that they follow the official MoonBit language specification and the
recommended examples.

Use this skill when you need to:

- Check whether MoonBit code follows the core language rules.
- Detect syntax issues, non-idiomatic patterns, or usage that diverges from the spec.
- Propose concrete, compilable fixes that are closer to official examples.
- Suggest small structural refactors that keep the same behavior but improve clarity.

Large‑scale refactors across many packages, or deep application‑level
architecture changes, are out of scope for this skill and should be handled
by a higher‑level design skill.

## What resources to rely on

When using this skill, you MUST rely on the MoonBit resources shipped with
the skill instead of inventing your own rules.

At minimum, the following resource files will be available:

- `resources/language-fundamentals.md`  
  Core concepts of MoonBit programs, top‑level declarations, and fundamentals.

- `resources/syntax-and-expressions.md`  
  Syntax of expressions, statements, functions, control flow, and other
  constructs relevant to everyday code.

- `resources/types-and-traits.md`  
  Type definitions, traits, methods, generics, and the rules that govern them.

- `resources/modules-and-packages.md`  
  Rules for modules, packages, visibility, imports, and project layout.

- `resources/examples-good.md`  
  Good, idiomatic MoonBit examples derived from the official `language`
  and `example` directories.

- `resources/examples-anti.md`  
  Non‑recommended or problematic patterns with improved alternatives.

If you are unsure which rule applies, FIRST search in the relevant resource
file above before making a judgement.

## How to use the resources

When evaluating code:

1. Start with fundamentals  
   - Skim `language-fundamentals.md` to recall the basic structure and
     constraints of MoonBit programs.
   - Ensure the submitted code respects these global rules before going into
     details.

2. Check syntax and control flow  
   - Use `syntax-and-expressions.md` to verify that all expressions,
     statements, function declarations, and control‑flow constructs conform
     to the spec.
   - Flag any syntax that appears invalid, ambiguous, or discouraged according
     to the documentation.

3. Validate types and traits when relevant  
   - If the code defines or heavily uses structs, enums, traits, methods, or
     generics, consult `types-and-traits.md`.
   - Check that trait implementations, method receivers, visibility, and
     generic usage follow the documented rules.
   - If there are better, more idiomatic ways to express the same type design,
     suggest them.

4. Check modules and packages for structure issues  
   - When the code spans multiple modules or packages, use
     `modules-and-packages.md` to validate imports, visibility, and layout.
   - Point out violations such as incorrect package names, broken import
     patterns, or structures that conflict with recommended layouts.

5. Compare against examples  
   - Use `examples-good.md` to find patterns that are similar to the user’s
     code, and guide your recommendations in that direction.
   - Use `examples-anti.md` to detect anti‑patterns; if the user code matches
     or is close to an anti‑pattern, explicitly mention the better alternative.

## Review and fix workflow

When a user asks you to review MoonBit code with this skill, follow this
workflow:

1. Clarify the scope  
   - Identify whether the user wants: (a) syntax/semantic correctness,
     (b) idiomatic style improvements, or (c) small structural refactors.
   - If not stated, assume they want both correctness and idiomatic style.

2. Summarize what the code appears to do  
   - In 1–3 sentences, restate the intended behavior, based on the code and
     any user description. This helps catch design mismatches.

3. Run a spec‑based check  
   - For each significant piece of the code, verify it against the relevant
     resource file(s) listed above.
   - Explicitly call out:
     - Violations of the documented rules.
     - Syntax that is invalid or questionable.
     - Usage that is legal but non‑idiomatic compared to examples.

4. Propose concrete fixes  
   - Provide fixed MoonBit code that:
     - Compiles according to the documented rules (as far as you can infer).
     - Preserves the original intent unless the user allows behavior changes.
     - Moves closer to the patterns from `examples-good.md`.
   - When you change code, explain briefly why the new version is more
     compliant or idiomatic.

5. Suggest follow‑ups (optional)  
   - If you notice deeper design issues (e.g., API boundary, module layout,
     trait design), briefly mention them and suggest that a higher‑level
     design skill or a separate review be used for those topics.

## What NOT to do

- Do NOT invent new MoonBit syntax or features that are not covered in the
  provided resources.
- Do NOT rely on error code explanations here; use the separate
  “MoonBit error lookup” skill for detailed error‑code diagnostics.
- Do NOT perform large, cross‑repository refactors or non‑local design
  changes. This skill focuses on local code review and small‑scale fixes.
- Do NOT silently change semantics without clearly explaining the impact.

## Example invocations

Use this skill when you receive prompts like:

- “/moonbit-evaluating-code この MoonBit の関数が言語仕様と公式の書き方に沿っているかレビューして、必要なら直して下さい。”
- “/moonbit-evaluating-code このモジュール構成と型定義が MoonBit として自然か、trait の切り方も含めてチェックしてほしいです。”
- “/moonbit-evaluating-code 公式 example に近い書き方になるように、このコードをリファクタしてください。”
- “/moonbit-evaluating-code 文法的には通るかもしれないが、MoonBit 的に変な書き方をしていないか教えてください。”

When responding, always ground your feedback in the documented rules and
examples from the resources directory, and make it clear which constraints
you are applying.
