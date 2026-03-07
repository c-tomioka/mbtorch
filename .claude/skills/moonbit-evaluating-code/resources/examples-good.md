# MoonBit Good Patterns and Idiomatic Examples

> Source: official MoonBit language reference and example projects
> (language/*, example/segment-tree, example/myers-diff, example/lambda)

This file collects idiomatic MoonBit patterns that should serve as positive
models when evaluating code. When reviewing MoonBit code, guide it toward
these patterns.

---

## 1. Enum + Pattern Matching for Data Modeling

Use enums with pattern matching instead of class hierarchies or tagged unions:

```moonbit
enum Node {
  Nil
  Node(Int, Node, Node)
}

fn Node::op_add(a : Node, b : Node) -> Node {
  match (a, b) {
    (Nil, _) => Nil
    (_, Nil) => Nil
    (Node(x, _, _), Node(y, _, _)) => Node(x + y, Nil, Nil)
  }
}
```

**Why**: This is MoonBit's natural way to express algebraic data types. The
compiler checks exhaustiveness, preventing missed cases.

---

## 2. ArrayView for Efficient Recursive Subdivision

Use `ArrayView` (slicing) instead of copying subarrays:

```moonbit
fn build(data : ArrayView[Int]) -> Node {
  if data.length() == 1 {
    Node(data[0], Nil, Nil)
  } else {
    let mid = data.length() / 2
    let left = build(data[0:mid])
    let right = build(data[mid:])
    left + right
  }
}

test "build tree" {
  let arr = [1, 2, 3, 4, 5]
  let tree = build(arr[:])
  inspect!(tree, content="Node(15, ...)")
}
```

**Why**: `ArrayView` avoids allocation overhead while maintaining clarity.

---

## 3. Functional Loop with `loop`/`continue`

Use the `loop` expression for tail-recursive-style iteration:

```moonbit
fn fib(n : Int) -> Int {
  loop n, 0, 1 {
    0, a, _b => a
    n, a, b => continue n - 1, b, a + b
  }
}
```

**Why**: The compiler can verify that `continue` passes the right number of
arguments, and the loop expression returns a value directly.

---

## 4. Method Definition with `TypeName::` Prefix

Always use the `TypeName::method_name` syntax for methods:

```moonbit
struct Point { x : Double; y : Double }

fn Point::distance(self : Point, other : Point) -> Double {
  let dx = self.x - other.x
  let dy = self.y - other.y
  (dx * dx + dy * dy).sqrt()
}

fn Point::origin() -> Point {
  { x: 0.0, y: 0.0 }
}
```

**Why**: The old `self`-parameter shorthand is deprecated. `TypeName::` makes
the association explicit and supports overloading.

---

## 5. Trait Implementation with `impl ... for ... with`

```moonbit
trait Drawable {
  draw(Self) -> String
}

impl Drawable for Point with draw(self) {
  "Point(\{self.x}, \{self.y})"
}
```

**Why**: This is the canonical syntax for trait implementations. Type annotations
can be omitted — the compiler infers them from the trait signature.

---

## 6. Operator Overloading via Traits

```moonbit
struct Vec2 { x : Double; y : Double }

impl Add for Vec2 with op_add(a, b) {
  { x: a.x + b.x, y: a.y + b.y }
}

impl Eq for Vec2 with op_equal(a, b) {
  a.x == b.x && a.y == b.y
}
```

**Why**: Clean, declarative syntax. Uses the builtin trait system.

---

## 7. Error Handling with `type!` and `raise`

```moonbit
type! ParseError {
  InvalidFormat(String)
  UnexpectedEof
}

fn parse_int(s : String) -> Int raise ParseError {
  if s.is_empty() { raise UnexpectedEof }
  // parsing logic...
  42
}

fn safe_parse(s : String) -> Int {
  try { parse_int(s) } catch {
    InvalidFormat(msg) => { println("Bad format: \{msg}"); 0 }
    UnexpectedEof => 0
  }
}
```

**Why**: Explicit error types in signatures, structured handling with `catch`.

---

## 8. `derive` for Common Trait Implementations

```moonbit
struct Config {
  name : String
  port : Int
  debug : Bool
} derive(Show, Eq, Hash, Default, ToJson, FromJson)
```

**Why**: Reduces boilerplate. All fields must implement the derived traits.

---

## 9. Guard for Early Exit

```moonbit
fn process(input : String?) -> String {
  guard let Some(s) = input else { return "no input" }
  guard s.length() > 0 else { return "empty" }
  "processed: \{s}"
}
```

**Why**: Flattens nested `if-let`/`match`. Makes the "happy path" prominent.

---

## 10. Test Blocks with Snapshot Testing

```moonbit
test "array operations" {
  let arr = [3, 1, 4, 1, 5]
  arr.sort()
  inspect!(arr, content="[1, 1, 3, 4, 5]")
}
```

Use `moon test --update` to auto-fill `content`. This keeps tests precise
and easy to maintain.

---

## 11. Constrained Generics

```moonbit
fn[T : Compare] max(a : T, b : T) -> T {
  if a.compare(b) > 0 { a } else { b }
}

fn[K : Hash + Eq, V] lookup(map : Map[K, V], key : K) -> V? {
  map.get(key)
}
```

**Why**: Type constraints are explicit and checked at compile time.

---

## 12. Labeled Arguments for Clarity

```moonbit
fn create_user(name~ : String, age~ : Int, admin~ : Bool = false) -> User {
  { name, age, admin }
}

// Clear at call site:
let user = create_user(name="Alice", age=30)
```

**Why**: Labeled arguments make function calls self-documenting, especially
when multiple parameters share the same type.

---

## 13. `for-in` with Iterators

```moonbit
fn sum(arr : Array[Int]) -> Int {
  let mut total = 0
  for x in arr {
    total = total + x
  }
  total
}

// Or with iterator methods:
fn sum2(arr : Array[Int]) -> Int {
  arr.iter().fold(init=0, fn(acc, x) { acc + x })
}
```

**Why**: `for-in` is concise and works with any iterable. Iterator methods
enable functional composition.

---

## 14. Struct Functional Update

```moonbit
fn with_debug(config : Config) -> Config {
  { ..config, debug: true }
}
```

**Why**: Creates a new struct without mutating the original. Clean and
intention-revealing.

---

## 15. Visibility Defaults (Encapsulation First)

```moonbit
// Internal representation hidden by default (abstract type)
struct Tensor {
  data : Array[Double]
  shape : Array[Int]
}

// Only expose what users need
pub fn Tensor::new(data : Array[Double], shape : Array[Int]) -> Tensor {
  { data, shape }
}

pub fn Tensor::shape(self : Tensor) -> Array[Int] {
  self.shape
}
```

**Why**: MoonBit defaults to abstract types. This is intentional — expose
the minimum API surface. Use `pub` or `pub(all)` only when genuinely needed.
