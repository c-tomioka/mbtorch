# MoonBit Types and Traits

> Source: official MoonBit language reference (language/fundamentals.md,
> language/methods.md, language/derive.md)

## Table of Contents

1. [Type Definitions](#type-definitions)
2. [Method System](#method-system)
3. [Operator Overloading](#operator-overloading)
4. [Trait System](#trait-system)
5. [Trait Objects](#trait-objects)
6. [Builtin Traits](#builtin-traits)
7. [Deriving Traits](#deriving-traits)

---

## Type Definitions

### Struct

```moonbit
struct Point {
  x : Double
  y : Double
} derive(Show, Eq)

// Mutable fields
struct Counter {
  mut count : Int
}

// Private fields in public struct
pub(all) struct Config {
  name : String
  priv secret : String  // hidden from outside
}
```

**Construction and update**:

```moonbit
let p = { x: 1.0, y: 2.0 }
let p2 = { ..p, x: 3.0 }   // functional update
```

### Enum

```moonbit
enum Shape {
  Circle(Double)
  Rectangle(Double, Double)
  Point
}

// Labeled fields in constructors
enum Expr {
  Literal(value~ : Int)
  BinOp(op~ : String, lhs~ : Expr, rhs~ : Expr)
}

// Construction with labels
let e = BinOp(op="+", lhs=Literal(value=1), rhs=Literal(value=2))
```

### Newtype

Wraps an existing type with a new identity:

```moonbit
type UserId Int
type Meters Double

// The inner value is accessible via .0
let id = UserId(42)
let raw = id.0  // 42
```

### Type Alias

Creates an alternative name (no new type):

```moonbit
typealias Name = String
```

---

## Method System

Methods are top-level functions associated with a type constructor.

### Defining Methods

**Recommended syntax** — prefix with `TypeName::`:

```moonbit
fn Point::distance(self : Point, other : Point) -> Double {
  let dx = self.x - other.x
  let dy = self.y - other.y
  (dx * dx + dy * dy).sqrt()
}

// Within method body, use Self to refer to the type:
fn Point::origin() -> Self {
  { x: 0.0, y: 0.0 }
}
```

**Deprecated syntax** — first parameter named `self` (may be removed in future):

```moonbit
// Not recommended for new code:
fn distance(self : Point, other : Point) -> Double { ... }
```

### Calling Methods

```moonbit
// Dot syntax (when first param is the type)
let d = p1.distance(p2)

// Qualified syntax
let d = Point::distance(p1, p2)
```

### Method Overloading

Different types can define methods of the same name:

```moonbit
fn Point::to_string(self : Point) -> String { ... }
fn Color::to_string(self : Color) -> String { ... }
// No conflict — each lives in its own namespace
```

### Local Methods

Private methods can be defined for foreign types locally (extension):

```moonbit
fn Int::double(self : Int) -> Int {
  self * 2
}
```

### Method Alias

```moonbit
#alias(dist)
fn Point::distance(self : Point, other : Point) -> Double { ... }
// Can now call: Point::dist(p1, p2)
```

---

## Operator Overloading

### Via Builtin Traits

| Operator | Trait     |
|----------|-----------|
| `+`      | `Add`     |
| `-`      | `Sub`     |
| `*`      | `Mul`     |
| `/`      | `Div`     |
| `%`      | `Mod`     |
| `==`     | `Eq`      |
| `<<`     | `Shl`     |
| `>>`     | `Shr`     |
| `-` (unary) | `Neg`  |
| `&`      | `BitAnd`  |
| `\|`     | `BitOr`   |
| `^`      | `BitXOr`  |

Example:

```moonbit
struct Vec2 { x : Double; y : Double }

impl Add for Vec2 with op_add(a, b) {
  { x: a.x + b.x, y: a.y + b.y }
}
```

### Via Method Alias (Indexing)

| Operator         | Alias      | Signature                                      |
|------------------|------------|-------------------------------------------------|
| `_[_]` (get)     | `"_[_]"`   | `(Self, Index) -> Result`                       |
| `_[_]=_` (set)   | `"_[_]=_"` | `(Self, Index, Value) -> Unit`                  |
| `_[_:_]` (view)  | `"_[_:_]"` | `(Self, start? : Index, end? : Index) -> Result`|

```moonbit
struct Grid { data : Array[Array[Int]] }

#alias("_[_]")
fn Grid::op_get(self : Grid, idx : (Int, Int)) -> Int {
  self.data[idx.0][idx.1]
}
```

---

## Trait System

### Declaring Traits

```moonbit
trait Printable {
  to_string(Self) -> String
}

// With default implementation (= _ marker required)
trait Greetable {
  name(Self) -> String
  greet(Self) -> String = _  // has default
}

impl Greetable with greet(self) {
  "Hello, \{Greetable::name(self)}!"
}
```

### Extending Traits (Super Traits)

```moonbit
trait Compare : Eq {
  compare(Self, Self) -> Int
}
```

To implement `Compare`, you must also implement `Eq`.

### Implementing Traits

```moonbit
impl Printable for Point with to_string(self) {
  "(\{self.x}, \{self.y})"
}

// Empty impl (when all methods have defaults, or to confirm compliance)
impl Greetable for User
```

### Using Traits (Constrained Generics)

```moonbit
fn[T : Eq] contains(arr : Array[T], elem : T) -> Bool {
  for x in arr {
    if x == elem { return true }
  }
  false
}
```

### Invoking Trait Methods Directly

```moonbit
fn[T : Printable] print_it(x : T) -> Unit {
  println(Printable::to_string(x))
}
```

### Dot Syntax for Trait Implementations

Trait implementations can be called via dot syntax with restrictions:
1. Regular methods are always favored over trait impls
2. Only trait impls in the package of the self type can use dot syntax
3. Multiple trait methods with same name from different traits cause ambiguity error

---

## Trait Objects

Runtime polymorphism via type erasure:

```moonbit
trait Animal {
  speak(Self) -> String
}

fn make_animals() -> Array[&Animal] {
  let dog = Dog::new()
  let cat = Cat::new()
  [dog as &Animal, cat as &Animal]
}
```

**Object safety** — methods must satisfy:
- `Self` must be the first parameter
- `Self` must appear only once in the method type (the first parameter)

New methods can be defined on trait objects:

```moonbit
fn &Animal::describe(self : &Animal) -> String {
  "An animal that says: \{self.speak()}"
}
```

---

## Builtin Traits

```moonbit
trait Eq {
  op_equal(Self, Self) -> Bool
}

trait Compare : Eq {
  compare(Self, Self) -> Int  // 0 equal, -1 smaller, 1 greater
}

trait Hash {
  hash_combine(Self, Hasher) -> Unit  // to be implemented
  hash(Self) -> Int                   // has default
}

trait Show {
  output(Self, Logger) -> Unit   // to be implemented
  to_string(Self) -> String      // has default
}

trait Default {
  default() -> Self
}
```

---

## Deriving Traits

MoonBit can automatically derive implementations for builtin traits.
All fields must implement the trait being derived.

```moonbit
struct User {
  name : String
  age : Int
} derive(Show, Eq, Compare, Hash, Default, ToJson, FromJson)
```

### Available Derives

| Derive       | Purpose                                    |
|--------------|--------------------------------------------|
| `Show`       | Pretty-printing (`to_string`)              |
| `Eq`         | Equality testing                           |
| `Compare`    | Ordering (fields compared in definition order) |
| `Default`    | Default values (structs: all defaults; enums: the parameterless case) |
| `Hash`       | Hashing for HashMap/HashSet                |
| `Arbitrary`  | Random value generation                    |
| `ToJson`     | JSON serialization                         |
| `FromJson`   | JSON deserialization                       |

### JSON Derive Options

```moonbit
// Enum style: "legacy" or "flat"
enum E { A; B(Int) } derive(ToJson(style="legacy"))

// Rename fields/cases
struct S { my_field : Int } derive(ToJson(rename_fields="camelCase"))
```

**Enum JSON styles**:
- `legacy`: `{ "$tag": "A" }`, `{ "$tag": "B", "0": 42 }`
- `flat`: `"A"`, `["B", 42]`

**Default derive for enums**: requires exactly one case with no parameters.
