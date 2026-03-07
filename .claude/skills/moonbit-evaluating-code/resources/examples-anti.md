# MoonBit Anti-Patterns

> Source: official MoonBit language reference and best practices

This file lists non-recommended patterns and their improved alternatives.
When reviewing MoonBit code that matches these anti-patterns, suggest the
corresponding fix.

---

## 1. Using `self` shorthand for method definition

**Anti-pattern** (deprecated):

```moonbit
fn distance(self : Point, other : Point) -> Double {
  ...
}
```

**Fix** — use `TypeName::` prefix:

```moonbit
fn Point::distance(self : Point, other : Point) -> Double {
  ...
}
```

**Why**: The `self`-parameter shorthand may be deprecated in the future. The
explicit prefix makes the type association clear and supports overloading.

---

## 2. Using `suberror A B` syntax

**Anti-pattern** (deprecated):

```moonbit
suberror MyError String
```

**Fix** — use `type!` with explicit constructors:

```moonbit
type! MyError {
  MyError(String)
}
```

**Why**: The old `suberror` syntax is officially deprecated.

---

## 3. Overusing `pub(all)` for types

**Anti-pattern**:

```moonbit
pub(all) struct InternalState {
  data : Array[Double]
  cache : Map[String, Int]
  dirty : Bool
}
```

**Fix** — use default (abstract) or `pub` (readonly) visibility:

```moonbit
struct InternalState {
  data : Array[Double]
  cache : Map[String, Int]
  mut dirty : Bool
}

pub fn InternalState::new() -> InternalState { ... }
pub fn InternalState::data(self : InternalState) -> Array[Double] {
  self.data
}
```

**Why**: MoonBit defaults to abstract types to encourage encapsulation. Only
use `pub(all)` for types that genuinely need full external construction and
mutation (like simple value objects or DTOs).

---

## 4. Ignoring exhaustiveness in match

**Anti-pattern**:

```moonbit
fn describe(shape : Shape) -> String {
  match shape {
    Circle(r) => "circle with radius \{r}"
    _ => "something else"  // hides missing cases
  }
}
```

**Fix** — enumerate all cases explicitly:

```moonbit
fn describe(shape : Shape) -> String {
  match shape {
    Circle(r) => "circle with radius \{r}"
    Rectangle(w, h) => "rectangle \{w}x\{h}"
    Point => "point"
  }
}
```

**Why**: Explicit patterns let the compiler catch new enum variants at compile
time. Wildcard `_` should only be used when truly handling an open set (like
`Error` matching).

---

## 5. Mutable variables instead of `loop`

**Anti-pattern**:

```moonbit
fn fib(n : Int) -> Int {
  let mut a = 0
  let mut b = 1
  let mut i = 0
  while i < n {
    let temp = b
    b = a + b
    a = temp
    i = i + 1
  }
  a
}
```

**Fix** — use `loop` expression:

```moonbit
fn fib(n : Int) -> Int {
  loop n, 0, 1 {
    0, a, _b => a
    n, a, b => continue n - 1, b, a + b
  }
}
```

**Why**: `loop` is more idiomatic, eliminates mutable state, and the compiler
can verify the structure.

---

## 6. Missing `raise` in function signature

**Anti-pattern**:

```moonbit
fn parse(input : String) -> Int {
  if input.is_empty() {
    abort("empty input")  // hidden runtime failure
  }
  ...
}
```

**Fix** — use proper error handling:

```moonbit
fn parse(input : String) -> Int raise ParseError {
  if input.is_empty() { raise ParseError::EmptyInput }
  ...
}
```

**Why**: `abort` crashes the program without recovery. Using `raise` makes
errors visible in the type signature and recoverable by callers.

---

## 7. Constructing `Failure` directly

**Anti-pattern**:

```moonbit
raise Failure("something went wrong")
```

**Fix** — use `fail()`:

```moonbit
fail!("something went wrong")
```

**Why**: `fail` includes source location automatically, producing better
error messages. It is officially preferred over direct `Failure` construction.

---

## 8. Copying arrays instead of using views

**Anti-pattern**:

```moonbit
fn process(arr : Array[Int], start : Int, end : Int) -> Int {
  let sub = arr.slice(start, end)  // allocates new array
  ...
}
```

**Fix** — use `ArrayView`:

```moonbit
fn process(arr : ArrayView[Int]) -> Int {
  ...
}

// Call with slice syntax:
process(arr[start:end])
```

**Why**: `ArrayView` avoids allocation and is the idiomatic way to work with
sub-sequences in MoonBit.

---

## 9. Missing `derive` for common traits

**Anti-pattern** — manual implementations of Show, Eq, etc.:

```moonbit
struct Point { x : Double; y : Double }

impl Show for Point with output(self, logger) {
  logger.write_string("Point(\{self.x}, \{self.y})")
}

impl Eq for Point with op_equal(a, b) {
  a.x == b.x && a.y == b.y
}
```

**Fix** — use `derive` when the default behavior is sufficient:

```moonbit
struct Point { x : Double; y : Double } derive(Show, Eq)
```

**Why**: `derive` is less error-prone and keeps code concise. Only implement
manually when custom behavior is needed.

---

## 10. Nested `if-let` instead of `guard`

**Anti-pattern**:

```moonbit
fn process(opt : Option[String]) -> String {
  if let Some(s) = opt {
    if s.length() > 0 {
      "result: \{s}"
    } else {
      "empty"
    }
  } else {
    "none"
  }
}
```

**Fix** — use `guard`:

```moonbit
fn process(opt : Option[String]) -> String {
  guard let Some(s) = opt else { return "none" }
  guard s.length() > 0 else { return "empty" }
  "result: \{s}"
}
```

**Why**: `guard` reduces nesting and makes the happy path prominent.

---

## 11. Implementing traits outside allowed packages

**Anti-pattern** — trying to implement a trait for a type from a different
package (when neither the type nor the trait is yours):

```moonbit
// In package C, trying to impl @a.Trait for @b.Type
impl @a.Drawable for @b.Widget with draw(self) { ... }
// ERROR: only @a or @b can define this impl
```

**Fix** — define a local trait or use a newtype wrapper:

```moonbit
// Option 1: local trait
trait MyDrawable {
  draw(Self) -> String
}
impl MyDrawable for @b.Widget with draw(self) { ... }

// Option 2: newtype wrapper
type DrawableWidget @b.Widget
impl @a.Drawable for DrawableWidget with draw(self) { ... }
```

**Why**: MoonBit's coherence rules ensure globally unique `Type: Trait`
implementations. This prevents silent conflicts across packages.

---

## 12. Forgetting to import core packages

**Anti-pattern** — using `@json` or `@test` without importing:

```moonbit
// Using @json.parse(...) but moon.pkg.json has no import for it
// Results in: core_package_not_imported warning
```

**Fix** — add to `import` in `moon.pkg.json`:

```json
{
  "import": [
    "moonbitlang/core/json"
  ]
}
```

**Why**: MoonBit requires explicit package imports. Even core packages must
be declared.

---

## 13. Hidden global mutable state

**Anti-pattern**:

```moonbit
// Top-level mutable state (not allowed directly)
// Workaround with Ref:
let counter : Ref[Int] = Ref::new(0)
```

**Fix** — pass state explicitly or use a struct:

```moonbit
struct Counter {
  mut value : Int
}

fn Counter::new() -> Counter { { value: 0 } }
fn Counter::increment(self : Counter) -> Unit { self.value = self.value + 1 }
```

**Why**: MoonBit intentionally disallows mutable top-level bindings. Global
mutable state makes code harder to reason about and test. When state is
necessary, make it explicit and contained.

---

## 14. Using `!!` (panic) in library code

**Anti-pattern**:

```moonbit
pub fn get_config() -> Config {
  load_config()!!  // panics if error, no recovery
}
```

**Fix** — propagate errors or provide fallback:

```moonbit
pub fn get_config() -> Config raise ConfigError {
  load_config()  // propagates error
}

// Or with fallback:
pub fn get_config_or_default() -> Config {
  try { load_config() } catch { _ => Config::default() }
}
```

**Why**: `!!` is suitable for tests and scripts, not for library code where
callers should decide how to handle errors.
