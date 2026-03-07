# MoonBit Language Fundamentals

> Source: official MoonBit language reference (language/introduction.md, language/fundamentals.md)

## Table of Contents

1. [Program Structure](#program-structure)
2. [Expressions and Statements](#expressions-and-statements)
3. [Variable Binding](#variable-binding)
4. [Naming Conventions](#naming-conventions)
5. [Keywords](#keywords)
6. [Program Entrance](#program-entrance)
7. [Built-in Data Types](#built-in-data-types)
8. [Functions](#functions)
9. [Control Flow](#control-flow)
10. [Pattern Matching](#pattern-matching)
11. [Iterators](#iterators)
12. [Overloaded Literals](#overloaded-literals)
13. [TODO Syntax](#todo-syntax)

---

## Program Structure

A MoonBit program consists of top-level definitions:

- type definitions
- function definitions
- constant definitions and variable bindings
- `init` functions, `main` function, and/or `test` blocks

---

## Expressions and Statements

MoonBit distinguishes between statements and expressions. In a function body,
only the last clause should be an expression, which serves as a return value.

```moonbit
fn foo() -> Int {
  let x = 1   // statement
  x + 1       // expression (return value)
}
```

**Expressions** include:
- Value literals (Boolean, numbers, characters, strings, arrays, tuples, structs)
- Arithmetical, logical, or comparison operations
- Array access `a[0]`, struct field `r.x`, tuple component `t.0`
- Variables and (capitalized) enum constructors
- Anonymous local function definitions
- `match`, `if`, `loop` expressions

**Statements** include:
- Named local function definitions
- Local variable bindings (`let`)
- Assignments
- `return` statements
- Any expression whose return type is `Unit` (e.g. `ignore`)

A code block can contain multiple statements and one expression; the value of
the expression is the value of the code block.

---

## Variable Binding

```moonbit
// Immutable binding
let x = 42

// Mutable binding
let mut i = 10
i = 20

// Top-level constant
const ZERO = 0

// Top-level let (requires explicit type annotation unless literal)
let zero = 0
```

Top-level variable bindings:
- Require **explicit** type annotation (unless defined using literals such as
  string, byte, or numbers)
- Cannot be mutable (use `Ref` instead)

---

## Naming Conventions

- **Variables, functions**: start with lowercase `a-z`, use `snake_case`
- **Constants, types**: start with uppercase `A-Z`, use `PascalCase` or `SCREAMING_SNAKE_CASE`

---

## Keywords

```
as, else, extern, fn, fnalias, if, let, const, match, using,
mut, type, typealias, struct, enum, trait, traitalias, derive,
while, break, continue, import, return, throw, raise, try, catch,
pub, priv, readonly, true, false, _, test, loop, for, in, impl,
with, guard, async, is, suberror, and, letrec, enumview, noraise, defer
```

---

## Program Entrance

### `init` and `main`

```moonbit
fn init {
  // No parameter list or return type
  // Can have multiple per package
  // Cannot be called explicitly
  // Runs during package initialization
  println("initializing")
}

fn main {
  // Main entrance of the program
  // Runs after initialization stage
  // Only in "main" packages
  println("running")
}
```

### `test` blocks

```moonbit
test "my test" {
  assert_eq!(1 + 1, 2)
}
```

Test blocks are essentially `() -> Unit!Error` functions.
If a test name starts with `"panic"`, it expects a panic to pass.

---

## Built-in Data Types

### Unit

`Unit` has only one value `()`. Used as return type for side-effecting
functions. Unlike `void` in other languages, it is a first-class type.

### Boolean

`true` and `false`. Negation: `!x` or `not(x)`.

### Number

| Type     | Description                  | Example          |
|----------|------------------------------|------------------|
| `Int16`  | 16-bit signed integer        | `(42 : Int16)`   |
| `Int`    | 32-bit signed integer        | `42`             |
| `Int64`  | 64-bit signed integer        | `1000L`          |
| `UInt`   | 32-bit unsigned integer      | `42U`            |
| `UInt64` | 64-bit unsigned integer      | `1000UL`         |
| `Double` | 64-bit floating point        | `3.14`           |
| `Float`  | 32-bit floating point        | `(1.0 : Float)`  |
| `BigInt` | Arbitrary precision integer  | `100000000000000000000N` |

Number prefixes: `0b`/`0B` (binary), `0o`/`0O` (octal), `0x`/`0X` (hex).
Underscores can be used as separators: `1_000_000`.

### String

UTF-16 code unit sequence. Multi-line strings with `#|`:

```moonbit
let s = "hello"
let multi =
  #|line 1
  #|line 2
```

String interpolation: `"x = \{x}"` (expression inside `\{...}`).

### Char

A single Unicode character: `'A'`, `'\n'`.

### Byte

A single byte literal: `b'A'`, `b'\xff'`.

### Bytes

A byte sequence literal: `b"hello"`.

### Tuple

Fixed-size collection of different types:

```moonbit
let t = (1, "hello", true)
let first = t.0  // access by index
```

### Array

Growable sequence of same-type elements:

```moonbit
let arr = [1, 2, 3]
let x = arr[0]
arr[1] = 42
```

`FixedArray` is a fixed-length array: `[1, 2, 3] : FixedArray[_]`.

### Map

Hash map literal:

```moonbit
let m = { "key1": 1, "key2": 2 }
```

### Option and Result

```moonbit
// Option
let some_val : Int? = Some(42)
let none_val : Int? = None

// Result
let ok_val : Result[Int, String] = Ok(42)
let err_val : Result[Int, String] = Err("error")
```

### Type Definitions

#### Struct

```moonbit
struct Point {
  x : Double
  y : Double
} derive(Show)

// Construction
let p = { x: 1.0, y: 2.0 }

// Field access
let px = p.x

// Functional update (creates new struct)
let p2 = { ..p, x: 3.0 }

// Mutable fields
struct MutablePoint {
  mut x : Double
  mut y : Double
}
```

#### Enum

```moonbit
enum Color {
  Red
  Green
  Blue
  Custom(Int, Int, Int)
}

let c = Red
let custom = Custom(255, 128, 0)
```

#### Newtype

Wraps an existing type with a new identity:

```moonbit
type UserId Int
type Meters Double
```

#### Type Alias

```moonbit
typealias Name = String
```

---

## Functions

### Basic Function Definition

```moonbit
fn add(x : Int, y : Int) -> Int {
  x + y
}
```

### Anonymous Functions (Closures)

```moonbit
let add = fn(x, y) { x + y }
// Short form for single-expression:
fn(x, y) { x + y }
```

### Labeled Arguments

```moonbit
fn greeting(name~ : String, greeting~ : String) -> String {
  "\{greeting}, \{name}!"
}

// Call with labels
greeting(name="Alice", greeting="Hello")

// Punning: if variable name matches label
let name = "Bob"
greeting(name~, greeting="Hi")
```

### Optional Arguments

```moonbit
fn greet(name~ : String, greeting~ : String = "Hello") -> String {
  "\{greeting}, \{name}!"
}

greet(name="Alice")  // uses default "Hello"
```

### Autofill Arguments

```moonbit
fn log(msg : String, loc~ : SourceLoc = _) -> Unit {
  println("\{loc}: \{msg}")
}
// SourceLoc is automatically filled by the compiler at call site
```

### Function Alias

```moonbit
fnalias add2 = add
```

---

## Control Flow

### If Expression

```moonbit
let x = if condition { 1 } else { 2 }

// If-let for pattern matching
if let Some(value) = optional {
  println(value)
}
```

### Match Expression

```moonbit
match value {
  0 => "zero"
  1 => "one"
  n => "other: \{n}"
}
```

### While Loop

```moonbit
while condition {
  // body
}

// While with continue condition
while i < 10, j < 20 {
  i = i + 1
  j = j + 2
}
```

### For Loop

```moonbit
for i = 0; i < 10; i = i + 1 {
  println(i)
}

// For-in loop (iterators)
for x in [1, 2, 3] {
  println(x)
}
```

### Loop Expression (Functional Loops)

```moonbit
fn fib(n : Int) -> Int {
  loop n, 0, 1 {
    0, _a, _b => _a
    n, a, b => continue n - 1, b, a + b
  }
}
```

### Guard Statement

```moonbit
guard condition else { return Err("failed") }

// Guard-let for pattern matching
guard let Some(x) = optional else { return None }
// x is now bound in the rest of the block
```

### Break and Return

`break` can return a value from a loop:

```moonbit
let result = for i = 0; i < 100; i = i + 1 {
  if arr[i] == target {
    break Some(i)
  }
} else {
  None
}
```

### Defer

```moonbit
fn read_file() -> String {
  let file = open("data.txt")
  defer { file.close() }
  file.read_all()
}
```

---

## Pattern Matching

### Supported Patterns

- **Literal**: `1`, `"hello"`, `true`
- **Variable**: `x` (binds matched value)
- **Wildcard**: `_` (matches anything, no binding)
- **Constructor**: `Some(x)`, `Ok(value)`
- **Tuple**: `(a, b, c)`
- **Struct**: `{ x, y }`, `{ x, .. }` (with rest)
- **Array**: `[a, b, ..]` (with rest)
- **Or-pattern**: `Red | Green | Blue`
- **As-pattern**: `Some(x) as opt`
- **Guard**: `x if x > 0`
- **Range**: `'a'..'z'`, `1..<10`

### `is` Expression

```moonbit
if value is Some(x) && x > 0 {
  println(x)
}
```

---

## Iterators

MoonBit provides `Iter[T]` and `Iter2[A, B]` for lazy iteration:

```moonbit
fn square(x : Iter[Int]) -> Iter[Int] {
  x.map(fn(v) { v * v })
}

[1, 2, 3].iter().filter(fn(x) { x > 1 }).each(fn(x) { println(x) })
```

Custom iterators use `Iter::new(fn(yield_))`:

```moonbit
fn my_range(start : Int, end : Int) -> Iter[Int] {
  Iter::new(fn(yield_) {
    for i = start; i < end; i = i + 1 {
      guard let IterContinue = yield_(i) else { x => break x }
    } else {
      IterContinue
    }
  })
}
```

---

## Overloaded Literals

MoonBit supports overloading integer, string, array, and map literals via
traits (`FromInt`, `FromString`, `FromArray`, etc.):

```moonbit
// If MyType implements FromInt:
let x : MyType = 42  // calls MyType::from_int(42)
```

---

## TODO Syntax

Placeholder for unfinished code. Compiles but panics at runtime:

```moonbit
fn todo_example() -> Int {
  ...  // TODO: implement
}
```
