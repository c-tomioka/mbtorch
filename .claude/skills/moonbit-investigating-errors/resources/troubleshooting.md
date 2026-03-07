# MoonBit Error Troubleshooting Guide

> Strategies, checklists, and decision trees for investigating MoonBit
> compiler errors.

## Table of Contents

1. [Quick Triage](#quick-triage)
2. [Error Categories and First Actions](#error-categories-and-first-actions)
3. [Cascading Error Strategy](#cascading-error-strategy)
4. [Common Patterns and Root Causes](#common-patterns-and-root-causes)
5. [Error Handling System Reference](#error-handling-system-reference)
6. [Build & Toolchain Issues](#build--toolchain-issues)

---

## Quick Triage

When faced with compiler output, follow this decision tree:

```
Error code present?
├── Yes → Look up in resources/index.md → Read error_codes/EXXXX.md
│         → Form hypothesis → Propose fix
└── No  → Is it a runtime error?
          ├── Yes → Check for panic, abort, or uncaught raise
          │         → Look at stack trace for function/line
          └── No  → Is it a build system error?
                    ├── Yes → Check moon.pkg.json, moon.mod.json
                    └── No  → Check toolchain installation
```

### Priority Rules

When multiple errors appear:

1. **Fix syntax errors (E3xxx) first** — they prevent parsing, which causes
   cascading type errors.
2. **Fix the earliest error** — errors later in the file often depend on
   earlier ones being resolved.
3. **Fix root-cause errors** — if error A causes error B, fix A first.
   Common cascading pairs:
   - Missing import (E4020) → unresolved type → type mismatch (E4014)
   - Wrong type definition → multiple method/trait errors
   - Missing `raise` in signature → unused try (E0023) or missing error handling

---

## Error Categories and First Actions

### E0xxx — Warnings

Warnings don't block compilation but signal potential issues.

**Unused items (E0001-E0009, E0015, E0031, E0032, E0037, E0046, E0052, E0053)**:
- Check if the item is genuinely unused or if there's a typo
- Make the item `pub` if it's part of the public API
- Remove the item if it's truly dead code
- Use `ignore()` to explicitly suppress if keeping intentionally

**Partial match (E0011)**:
- Add missing patterns to cover all enum variants
- This is often a real bug — the compiler catches missing cases

**Deprecated (E0020, E0027)**:
- Read the deprecation message for migration guidance
- Common: `self` parameter shorthand → `TypeName::method_name` syntax
- Common: `suberror A B` → `type! A { A(B) }`

**Type variable issues (E0005, E0013)**:
- E0005: remove unused generic parameter or use it in the function body
- E0013: add explicit type annotations to resolve ambiguity

### E3xxx — Syntax Errors

These block compilation entirely. Common fixes:

**E3001 (lexing error)**:
- Check for unclosed strings, chars, or comments
- Look for non-ASCII characters that shouldn't be there
- Verify string interpolation syntax: `\{expr}` not `${expr}`

**E3002 (parse error)**:
- Check for missing brackets, parentheses, or braces
- Verify function syntax: `fn name(params) -> ReturnType { body }`
- Check `match` arm syntax: `pattern => expr`

**E3003 (init/main signature)**:
- `fn init` and `fn main` cannot have parameters or return types
- Remove any `()` parameter list and `-> Type` return annotation

**E3010 (default values)**:
- Only labelled arguments (with `~`) can have defaults
- Change `fn f(x : Int = 0)` to `fn f(x~ : Int = 0)`

**E3020 (= vs : in structs)**:
- Struct construction uses `:` not `=`
- Change `{ field = value }` to `{ field: value }`

**E3800 (comma vs semicolon/newline)**:
- In enum/struct definitions, use newlines or `;` to separate fields
- Do NOT use commas — this is a very common mistake from other languages

### E4xxx — Type/Semantic Errors

These are the most varied. Key patterns:

**E4014 (type mismatch)** — most common error:
- Check return type matches function signature
- Check argument types at call sites
- Look for implicit Unit return when non-Unit expected
- Remember: last expression in a block is the return value

**E4015 (no such method)**:
- Verify the type has the method defined
- Check if the method is defined with `TypeName::method` syntax
- Ensure the package defining the method is imported
- Check spelling

**E4018 (trait not implemented)**:
- Add `impl Trait for Type with method(...)` for missing implementations
- Check if all required trait methods are implemented
- For sealed traits (`pub trait` without `open`), only the defining
  package can add implementations

**E4020 (package not found)**:
- Add the package to `import` in `moon.pkg.json`
- Check the package path is correct
- Run `moon update` if it's an external dependency

---

## Cascading Error Strategy

Some errors cause many downstream errors. Recognizing these saves time:

### Pattern 1: Missing Import Cascade

```
E4020: Package not found
  → E4014: Type mismatch (unknown type becomes error)
  → E4015: No such method (methods from missing package)
```

**Fix**: Add the import. All downstream errors typically resolve.

### Pattern 2: Type Definition Error Cascade

```
Wrong struct/enum definition
  → E4014: Type mismatches everywhere the type is used
  → E4015: Methods don't resolve
  → E4018: Trait implementations fail
```

**Fix**: Fix the type definition first, rebuild.

### Pattern 3: Error Handling Cascade

```
Missing `raise` in function signature
  → E0023: Unused try in caller
  → E4014: Return type mismatch (Result vs plain type)
```

**Fix**: Add `raise ErrorType` to the function signature.

### Pattern 4: Visibility Cascade

```
Type is abstract (default) but used externally
  → E4001: Visibility mismatch
  → Construction fails in other packages
  → Pattern matching fails in other packages
```

**Fix**: Change visibility to `pub` (readonly) or `pub(all)` (full access).

---

## Common Patterns and Root Causes

### "My code used to work but now it doesn't"

1. Check for MoonBit version updates (syntax changes, deprecated features)
2. Check for dependency updates (`moon update`)
3. Look for E0027 (deprecated_syntax) warnings that became errors

### "I copied code from a tutorial and it doesn't compile"

1. Tutorial may use outdated syntax
2. Check for `self` parameter shorthand (deprecated) → use `TypeName::method`
3. Check for `suberror` syntax (deprecated) → use `type!`
4. Verify all imports are declared in `moon.pkg.json`

### "The error message doesn't match what I see in my code"

1. Clean build: `moon clean && moon build`
2. Check if you're editing the right file
3. Look for macro-generated code or derive-generated code
4. Check if the error is in a dependency, not your code

### "I get a type mismatch but the types look the same"

1. Check for same-name types from different packages
2. Check for `pub` vs `pub(all)` — readonly types can't be constructed outside
3. Check for generic type parameter mismatches
4. Check if `Option[T]` vs `T` or `Result[T, E]` vs `T`

---

## Error Handling System Reference

MoonBit has a structured error handling system. Errors in this area are common:

### Error Type Definition

```moonbit
// Define custom error types
type! MyError {
  NotFound(String)
  InvalidInput
}

// WRONG (deprecated):
// suberror MyError String
```

### Function Signatures with Errors

```moonbit
// Function that may raise a specific error
fn parse(s : String) -> Int raise MyError { ... }

// Function that may raise any error
fn risky() -> Int raise { ... }

// Function that never raises
fn safe() -> Int noraise { ... }
```

### Handling Errors

```moonbit
// Rethrow (default behavior)
let x = parse(input)  // rethrows if error

// Try-catch
let x = try { parse(input) } catch {
  NotFound(msg) => { println(msg); 0 }
  InvalidInput => 0
}

// Convert to Result
let result : Result[Int, MyError] = try? { parse(input) }

// Panic on error (for tests/scripts, not libraries)
let x = parse(input)!!
```

### Common Error Handling Mistakes

| Mistake | Error Code | Fix |
|---------|-----------|-----|
| Missing `raise` in signature | E4014 (type mismatch) | Add `raise ErrorType` |
| Using `raise` without error type defined | E3002 | Define `type! MyError { ... }` |
| `try` on non-raising function | E0023 | Remove `try` |
| Calling raising function without handling | E4014 | Add `raise` to caller or use `try`/`catch` |
| Pattern matching `Error` without `_` | E0011 | Add wildcard `_` branch |

---

## Build & Toolchain Issues

### moon.pkg.json Problems

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| E3017 | Invalid JSON in moon.pkg.json | Validate JSON syntax |
| E4020 | Package not in `import` | Add to `import` array |
| E0029 | Package imported but not used | Remove from `import` |
| "module not found" | Missing `deps` in moon.mod.json | Add to `deps` |

### Common moon.pkg.json Structure

```json
{
  "import": [
    "moonbitlang/core/json",
    { "path": "user/project/utils", "alias": "u" }
  ],
  "test-import": [
    "moonbitlang/core/json"
  ]
}
```

### Test File Issues

| File Pattern | Type | Imports |
|-------------|------|---------|
| `*_test.mbt` | BlackBox test | Only public API + `test-import` |
| `*_wbtest.mbt` | WhiteBox test | All members + `wbtest-import` |
| Other `.mbt` | Source code | `import` only |

If a test can't access a function, check:
1. Is the function `pub`? (BlackBox tests need public API)
2. Is the test in the right file type? (`_test.mbt` vs `_wbtest.mbt`)
3. Are test dependencies in `test-import` or `wbtest-import`?
