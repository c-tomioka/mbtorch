# MoonBit Syntax and Expressions

> Source: official MoonBit language reference (language/fundamentals.md,
> language/error-handling.md, language/tests.md, language/benchmarks.md,
> language/attributes.md, language/docs.md, language/ffi.md)

## Table of Contents

1. [Error Handling](#error-handling)
2. [Testing](#testing)
3. [Benchmarks](#benchmarks)
4. [Attributes](#attributes)
5. [Documentation](#documentation)
6. [FFI (Foreign Function Interface)](#ffi)
7. [Async (Experimental)](#async)

---

## Error Handling

### Error Types

All error values are represented by the `Error` type. Concrete error types
must be defined:

```moonbit
// Enum-style error type
type! DivError {
  DivByZero
}

// With payload
type! FileError {
  NotFound(String)
  PermissionDenied(String)
}

// Builtin error: Failure
// Use fail("message") to raise it (preferred over constructing Failure directly)
```

The older `suberror A B` syntax is **deprecated**. Use `suberror A { A(B) }` instead.

### Throwing Errors

Use `raise` in function signature to indicate the function may throw:

```moonbit
fn div(x : Int, y : Int) -> Int raise DivError {
  if y == 0 { raise DivByZero }
  x / y
}

// Omit error type to use generic Error:
fn might_fail() -> Int raise { ... }

// Function that never raises:
fn safe() -> Int noraise { 42 }
```

### Handling Errors

**Direct call** (rethrows automatically):

```moonbit
fn caller() -> Int raise DivError {
  div(10, 0)  // rethrows if error
}
```

**Try-catch**:

```moonbit
try {
  div(10, 0)
} catch {
  DivByZero => println("cannot divide by zero")
} noraise {
  // executed when no error
}
```

When the body of `try` is a simple expression, `try` keyword can be omitted:

```moonbit
let result = div(10, 2) catch { DivByZero => 0 }
```

**Transform to Result**:

```moonbit
let result : Result[Int, DivError] = try? { div(10, 0) }
```

**Panic on error**:

```moonbit
let value = div(10, 2)!!  // panics if error
```

### Error Polymorphism

Use `raise?` to indicate a function may or may not throw depending on its
parameters:

```moonbit
fn map_poly[A, B](arr : Array[A], f : (A) -> B raise?) -> Array[B] raise? {
  ...
}
```

---

## Testing

### Test Blocks

```moonbit
test "descriptive name" {
  assert_eq!(1 + 1, 2)
}
```

- A test block is `() -> Unit!Error`
- Test name starting with `"panic"` expects a panic:

```moonbit
test "panic division by zero" {
  let _ = div(1, 0)!!
}
```

### Snapshot Tests

**Inspect (Show)**:

```moonbit
test "inspect example" {
  inspect!([1, 2, 3], content="[1, 2, 3]")
}
```

Use `moon test --update` to auto-fill `content`.

**JSON inspect**:

```moonbit
test "json inspect" {
  @json.inspect!(my_struct, content={"field": 42})
}
```

**Full snapshot**:

```moonbit
test "full snapshot" (it : @test.T) {
  it.writeln("output line 1")
  it.writeln("output line 2")
  it.snapshot!(filename="my_snapshot.txt")
}
```

Creates files under `__snapshot__/`.

### BlackBox vs WhiteBox Tests

- **WhiteBox** (`_wbtest.mbt`): access to all members in the package
- **BlackBox** (`_test.mbt`): access only to public members

WhiteBox tests import packages from `import` and `wbtest-import`.
BlackBox tests import current package + packages from `import` and `test-import`.

---

## Benchmarks

```moonbit
test "benchmark fib" (b : @bench.T) {
  b.bench(fn() { fib(20) })
}
```

- `@bench.T::keep` prevents optimizing away pure computations
- `name` parameter for batch comparison:

```moonbit
test "compare" (b : @bench.T) {
  b.bench(fn() { naive_fib(20) }, name="naive")
  b.bench(fn() { fast_fib(20) }, name="fast")
}
```

---

## Attributes

Attributes are annotations: `#attribute(...)`. They occupy entire lines.

### Common Attributes

| Attribute | Purpose |
|-----------|---------|
| `#deprecated("msg")` | Mark API as deprecated |
| `#alias("_[_]")` | Operator overloading for indexing |
| `#alias(new_name)` | Create function alias |
| `#internal(category, "msg")` | Mark as internal (warns on cross-module use) |
| `#external` | Mark type as external (anyref/any/void*) |
| `#cfg(target="wasm")` | Conditional compilation |
| `#skip` | Skip a test block |
| `#visibility(change_to="readonly")` | Hint future visibility change |
| `#warnings("-unused_value")` | Configure warnings for a declaration |
| `#borrow(params)` | FFI: borrow calling convention |
| `#as_free_fn` | Declare method also as free function |
| `#callsite(autofill(loc))` | Auto-fill SourceLoc at call site |
| `#label_migration(...)` | Help evolve API labels |

### Deprecated Attribute Forms

```moonbit
#deprecated
#deprecated("Use new_function instead")
#deprecated("msg", skip_current_package=true)
```

### Alias Attribute Forms

```moonbit
// Operator overloading
#alias("_[_]")
fn MyContainer::op_get(self : MyContainer, idx : Int) -> Int { ... }

// Named alias with visibility
#alias(old_name, visibility="pub", deprecated="Use new_name")
fn new_name() -> Unit { ... }
```

---

## Documentation

Doc comments use `///` prefix (markdown):

```moonbit
/// Adds two integers.
///
/// # Examples
///
/// ```mbt check
/// test { assert_eq!(add(1, 2), 3) }
/// ```
fn add(x : Int, y : Int) -> Int { x + y }
```

- `mbt check` code blocks are treated as document tests
- Doc tests are always blackbox tests
- Private definitions cannot have document tests

---

## FFI

### Backends

MoonBit has five backends: Wasm, Wasm GC, JavaScript, C, LLVM (experimental).

### Declaring Foreign Functions

**Wasm (import from host)**:

```moonbit
fn cos(d : Double) -> Double = "math" "cos"
```

**Wasm (inline)**:

```moonbit
extern "wasm" fn identity(d : Double) -> Double =
  #|(func (param f64) (result f64))
```

**JavaScript (import)**:

```moonbit
fn cos(d : Double) -> Double = "Math" "cos"
```

**JavaScript (inline)**:

```moonbit
extern "js" fn cos(d : Double) -> Double =
  #|(d) => Math.cos(d)
```

**C**:

```moonbit
extern "C" fn put_char(ch : UInt) = "function_name"
```

### External Types

```moonbit
#external
type Ptr  // anyref (Wasm), any (JS), void* (C)
```

### Callbacks

`FuncRef[T]` represents a closed function (no captures):

```moonbit
let f : FuncRef[(Int) -> Int] = fn(x) { x + 1 }
```

### Constant Enum Integer Values

```moonbit
enum Flags {
  None = 0
  Read = 1
  Write = 2
  ReadWrite = 3
}
```

### Exporting Functions

Configure `exports` in `moon.pkg.json`:

```json
{ "link": { "<backend>": { "exports": ["add", "fib:test"] } } }
```

---

## Async

Async functions use the `async` keyword. They implicitly raise errors
and require `noraise` if they don't.

```moonbit
async fn fetch_data(url : String) -> String raise {
  @http.get(url)
}
```

- Async functions can only be called inside async functions
- Use `async fn main` and `async test` for entry points
- Requires `moonbitlang/async` package
- Structured concurrency via `@async.with_task_group`
- Currently best supported on native backend
