# MoonBit Error Codes Index

> Source: official MoonBit language reference (`docs/moonbit-docs/error_codes/`)

This index organizes all MoonBit compiler error codes by category for quick
lookup during error investigation. Individual error code details are in
`error_codes/EXXXX.md`.

## Error Code Ranges

| Range | Category | Severity |
|-------|----------|----------|
| E0xxx | Warnings | Non-fatal (code compiles but may have issues) |
| E1xxx | Internal Compiler Errors (ICE) | Fatal (compiler bug) |
| E3xxx | Syntax / Parse Errors | Fatal (code doesn't parse) |
| E4xxx | Type Checking / Semantic Errors | Fatal (code doesn't type-check) |

---

## E0xxx — Warnings

### Unused Items

| Code | Warning Name | Description |
|------|-------------|-------------|
| E0001 | unused_value | Unused function |
| E0002 | unused_value | Unused variable |
| E0003 | unused_type_declaration | Unused type declaration |
| E0006 | unused_constructor | Enum variant never used |
| E0007 | unused_field | Struct/enum field never read |
| E0009 | struct_never_constructed | Struct never instantiated |
| E0015 | unused_mut | `mut` declared but never mutated |
| E0023 | unused_try | `try` body never raises errors |
| E0024 | unused_error_type | Error type in signature never raised |
| E0031 | unused_optional_argument | Optional argument never supplied |
| E0032 | unused_default_value | Default value of optional arg never used |
| E0037 | unused_loop_label | Loop label never referenced |
| E0046 | unused_rest_mark | Useless `..` in pattern (all fields matched) |
| E0049 | missing_definition | Pub definition not in `.mbti` file |
| E0052 | unused_loop_variable | Loop variable not updated in loop body |
| E0053 | unused_trait_bound | Trait bound never exercised |
| E0058 | unused_non_capturing | Unnecessary non-capturing group in regex |
| E0060 | unused_struct_update | Struct update result unused |
| E0061 | duplicate_test | Two tests share the same name |
| E0067 | unused_async | Redundant `async` annotation |

### Code Quality / Logic

| Code | Warning Name | Description |
|------|-------------|-------------|
| E0004 | missing_priv | Abstract type not in public API, should be `priv` |
| E0005 | unused_type_variable | Generic type variable unused |
| E0008 | redundant_modifier | Redundant visibility modifier |
| E0011 | partial_match | Pattern matching doesn't cover all cases |
| E0012 | unreachable_code | Code after return or exhaustive match |
| E0013 | unresolved_type_variable | Type has unresolved type variables |
| E0017 | ambiguous_loop_argument | Ambiguous identifier in loop |
| E0018 | useless_loop | `loop` has no `continue` |
| E0021 | missing_pattern_arguments | Constructor args omitted without `..` |
| E0022 | ambiguous_block | Ambiguous `{ value }` (block or struct?) |
| E0028 | todo | Unfinished code (`...`) |
| E0036 | loop_label_shadowing | Loop label shadows outer label |
| E0041 | missing_rest_mark | Closed map pattern missing `..` |
| E0050 | method_shadowing | Local method shadows upstream method |
| E0051 | ambiguous_precedence | Ambiguous operator precedence |
| E0056 | missing_pattern_field | Missing field in struct pattern |
| E0057 | missing_pattern_payload | Constructor pattern missing payload |
| E0062 | invalid_cascade | Non-unit return with `..` cascade |
| E0063 | syntax_lint | Code pattern discouraged by linter |

### Deprecated & Migration

| Code | Warning Name | Description |
|------|-------------|-------------|
| E0014 | alert | Usage of `#internal`-marked API |
| E0020 | deprecated | Deprecated API usage |
| E0025 | test_unqualified_package | Test uses implicitly imported API |
| E0026 | unused_catch_all | Complete patterns but `catch!` used |
| E0027 | deprecated_syntax | Deprecated syntax form |
| E0029 | unused_package | Unused package in dependencies |
| E0030 | missing_package_alias | Package alias is empty |
| E0035 | reserved_keyword | Word reserved for future use |
| E0042 | invalid_attribute | Unrecognized/malformed attribute |
| E0043 | unused_attribute | Attribute has no effect |
| E0068 | declaration_unimplemented | Declaration left unimplemented |
| E0069 | declaration_implemented | Declaration already implemented |

### FFI & Backend

| Code | Warning Name | Description |
|------|-------------|-------------|
| E0044 | invalid_inline_wasm | Inline wasm refers to unbound function |
| E0055 | unannotated_ffi | FFI param missing `#borrow`/`#owned` |
| E0059 | unaligned_byte_access | Unaligned byte access in bits pattern |

### Array / Data Structure Hints

| Code | Warning Name | Description |
|------|-------------|-------------|
| E0064 | unannotated_toplevel_array | Toplevel array literal lacks type annotation |
| E0065 | prefer_readonly_array | Array used read-only, prefer `ReadOnlyArray` |
| E0066 | prefer_fixed_array | Array mutated, prefer `FixedArray` |

### Not Emitted / Internal

| Code | Description |
|------|-------------|
| E0016 | Parser consistency check (not emitted) |
| E0019 | Toplevel not left-aligned (not emitted) |
| E0033 | Text segment exceeds limits |
| E0034 | Implicit use of builtin definitions |
| E0038 | Loop missing `invariant` clause |
| E0039 | Loop missing `reasoning` clause |
| E0040, E0045, E0054 | No longer emitted |
| E0047 | Invalid `.mbti` file |

---

## E1xxx — Internal Compiler Errors

| Code | Description |
|------|-------------|
| E1001 | Internal compiler error (ICE). File a bug report. |

---

## E3xxx — Syntax / Parse Errors

| Code | Description |
|------|-------------|
| E3001 | Invalid/incomplete tokens (lexing error) |
| E3002 | Syntax error (general parse error) |
| E3003 | `init`/`main` must have no args and no return type |
| E3004 | Missing parameter list (add `()`) |
| E3005 | Invalid visibility for entity |
| E3006 | No individual visibility on enum constructors |
| E3007 | `..` must be at end of pattern |
| E3008 | Multiple `..` in array pattern |
| E3009 | Struct pattern cannot be only `..` (use `_`) |
| E3010 | Only labelled args can have default values |
| E3011 | Invalid left-hand-side in assignment |
| E3012 | Struct pattern and map pattern cannot mix |
| E3014 | Inline wasm syntax error |
| E3016 | Unexpected `~` in argument |
| E3017 | JSON parse error in `moon.pkg.json` |
| E3018 | Range pattern bounds must be constant |
| E3019 | Inclusive range `a..=b` cannot have `_` as upper bound |
| E3020 | Unexpected `=` in struct expression (use `:`) |
| E3800 | Expecting newline/`;` in enum/struct (not `,`) |

---

## E4xxx — Type Checking / Semantic Errors

### Type & Visibility

| Code | Description |
|------|-------------|
| E4000 | Duplicate generic type variable name |
| E4001 | Incompatible field visibility |
| E4002 | Unsupported visibility modifier |
| E4003 | Reserved type name (e.g., `Error`) |
| E4014 | Type mismatch |

### Trait & Method

| Code | Description |
|------|-------------|
| E4004 | Trait methods cannot have type parameters |
| E4005 | Duplicate method name in trait |
| E4010 | `pub` not allowed on default trait impl |
| E4011 | Type params not allowed on default trait impl |
| E4015 | Type has no method with given name |
| E4017 | Ambiguous method (multiple traits) |
| E4018 | Type doesn't implement required trait |

### Function & Parameter

| Code | Description |
|------|-------------|
| E4006 | Duplicate local function name |
| E4007 | Enum variant without payload cannot be called as function |
| E4008 | FFI function cannot have type parameters |
| E4009 | Incoherent argument count in match branches |
| E4012 | Mutable constructor fields only on labelled args |
| E4013 | Wrong number of function arguments |
| E4019 | Duplicate label in function |

### Module & Package

| Code | Description |
|------|-------------|
| E4020 | Package not found in loaded packages |

### Advanced (selected high-frequency)

| Code | Description |
|------|-------------|
| E4100 | Not a trait (used where trait expected) |
| E4151 | `FuncRef[T]` must be closed function |

For the full list of E4xxx codes (100+ entries), look up individual
`error_codes/E4xxx.md` files.
