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

Large-scale refactors across many packages, or deep application-level
architecture changes, are out of scope for this skill and should be handled
by a higher-level design skill.

## What resources to rely on

When using this skill, you MUST rely on the MoonBit resources shipped with
the skill instead of inventing your own rules.

All resource files are derived from the **official MoonBit language reference**
(`docs/moonbit-docs/language/`) and the official example projects
(`docs/moonbit-docs/example/`). They are optimized for quick lookup during
code evaluation.

- `resources/language-fundamentals.md`
  Program structure, expressions vs statements, variable binding, naming
  conventions, keywords, program entrance (`init`/`main`/`test`), all
  built-in data types, functions (labeled/optional/autofill args), control
  flow (`if`, `match`, `while`, `for`, `loop`, `guard`, `defer`), pattern
  matching, iterators, overloaded literals, TODO syntax.

- `resources/syntax-and-expressions.md`
  Error handling (`type!`, `raise`, `try`/`catch`, `try?`, `!!`, error
  polymorphism), testing (test blocks, snapshot tests, blackbox/whitebox),
  benchmarks, attributes (`#deprecated`, `#alias`, `#cfg`, etc.),
  documentation (`///` doc comments, doc tests), FFI (Wasm/JS/C backends,
  foreign types/functions, callbacks, exports), async (experimental).

- `resources/types-and-traits.md`
  Struct, enum (with labeled fields), newtype, type alias. Method system
  (`TypeName::` prefix, dot syntax, local methods, method alias). Operator
  overloading (via traits and `#alias`). Trait system (declaration, super
  traits, `impl ... for ... with`, default impls, constrained generics,
  trait invocation). Trait objects. Builtin traits (`Eq`, `Compare`, `Hash`,
  `Show`, `Default`). Deriving (`Show`, `Eq`, `Compare`, `Default`, `Hash`,
  `Arbitrary`, `ToJson`/`FromJson` with options).

- `resources/modules-and-packages.md`
  Packages vs modules, `moon.mod.json` and `moon.pkg.json` configuration,
  imports and default aliases, `using` syntax, internal packages. Access
  control for functions, types (`priv`/abstract/`pub`/`pub(all)`), and
  traits (`priv`/abstract/`pub`/`pub(open)`). Trait implementation location
  rules (coherence). Virtual packages (experimental).

- `resources/examples-good.md`
  15 idiomatic patterns: enum + pattern matching, ArrayView subdivision,
  functional `loop`, `TypeName::` methods, `impl ... for ... with` traits,
  operator overloading, error handling, `derive`, `guard`, snapshot testing,
  constrained generics, labeled arguments, `for-in` iterators, functional
  struct update, visibility defaults.

- `resources/examples-anti.md`
  14 anti-patterns with fixes: deprecated `self` shorthand, deprecated
  `suberror` syntax, overusing `pub(all)`, wildcard in match, mutable vars
  instead of `loop`, missing `raise`, direct `Failure` construction, array
  copying instead of views, manual trait impls vs `derive`, nested `if-let`
  vs `guard`, cross-package trait impl violations, missing core imports,
  hidden global state, `!!` in library code.

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
     statements, function declarations, and control-flow constructs conform
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
   - Use `examples-good.md` to find patterns that are similar to the user's
     code, and guide your recommendations in that direction.
   - Use `examples-anti.md` to detect anti-patterns; if the user code matches
     or is close to an anti-pattern, explicitly mention the better alternative.

## Review and fix workflow

When a user asks you to review MoonBit code with this skill, follow this
workflow:

1. Clarify the scope
   - Identify whether the user wants: (a) syntax/semantic correctness,
     (b) idiomatic style improvements, or (c) small structural refactors.
   - If not stated, assume they want both correctness and idiomatic style.

2. Summarize what the code appears to do
   - In 1-3 sentences, restate the intended behavior, based on the code and
     any user description. This helps catch design mismatches.

3. Run a spec-based check
   - For each significant piece of the code, verify it against the relevant
     resource file(s) listed above.
   - Explicitly call out:
     - Violations of the documented rules.
     - Syntax that is invalid or questionable.
     - Usage that is legal but non-idiomatic compared to examples.

4. Propose concrete fixes
   - Provide fixed MoonBit code that:
     - Compiles according to the documented rules (as far as you can infer).
     - Preserves the original intent unless the user allows behavior changes.
     - Moves closer to the patterns from `examples-good.md`.
   - When you change code, explain briefly why the new version is more
     compliant or idiomatic.

5. Suggest follow-ups (optional)
   - If you notice deeper design issues (e.g., API boundary, module layout,
     trait design), briefly mention them and suggest that a higher-level
     design skill or a separate review be used for those topics.

## What NOT to do

- Do NOT invent new MoonBit syntax or features that are not covered in the
  provided resources.
- Do NOT rely on error code explanations here; use the separate
  "MoonBit error lookup" skill for detailed error-code diagnostics.
- Do NOT perform large, cross-repository refactors or non-local design
  changes. This skill focuses on local code review and small-scale fixes.
- Do NOT silently change semantics without clearly explaining the impact.

## Example invocations

Use this skill when you receive prompts like:

- "/moonbit-evaluating-code Check if this MoonBit function follows the language spec and fix it if needed."
- "/moonbit-evaluating-code Review this module structure and type definitions for idiomatic MoonBit, including trait design."
- "/moonbit-evaluating-code Refactor this code to be closer to official MoonBit examples."
- "/moonbit-evaluating-code Is there anything non-idiomatic or spec-violating in this MoonBit code?"

When responding, always ground your feedback in the documented rules and
examples from the resources directory, and make it clear which constraints
you are applying.
