# MoonBit Modules and Packages

> Source: official MoonBit language reference (language/packages.md)

## Table of Contents

1. [Packages and Modules](#packages-and-modules)
2. [Package Configuration](#package-configuration)
3. [Importing and Using Packages](#importing-and-using-packages)
4. [Internal Packages](#internal-packages)
5. [Access Control](#access-control)
6. [Virtual Packages](#virtual-packages)

---

## Packages and Modules

### Package

The most important unit for code organization. A package consists of:
- Source code files (`*.mbt`)
- A single package configuration file (`moon.pkg.json` or `moon.pkg`)

A package can be:
- A **main package** (contains a `main` function, marked with `is-main`)
- A **library package** (provides code for other packages)

### Module

A project corresponds to a module. It consists of:
- Multiple packages
- A single `moon.mod.json` configuration file

**Identification**:
- Module: `user-name/project-name` (the `name` field in `moon.mod.json`)
- Package: `user-name/project-name/path-to-pkg` (relative path from source root)

---

## Package Configuration

### Dependencies

**Module-level** — declare in `moon.mod.json`:

```json
{
  "name": "user/project",
  "deps": {
    "moonbitlang/core": "0.1.0"
  }
}
```

**Package-level** — declare in `moon.pkg.json` or `moon.pkg`:

```json
{
  "import": [
    "moonbitlang/core/json",
    { "path": "user/project/utils", "alias": "u" }
  ]
}
```

In `moon.pkg` format:

```
import(
  "moonbitlang/core/json"
  "user/project/utils" as u
)
```

**Important**: if you use `@json`, `@test`, or other core aliases, add the
corresponding `moonbitlang/core/...` package to `import` to avoid
`core_package_not_imported` warnings.

### Default Alias

The **default alias** of a package is the last segment of its path split
by `/`. For example, `moonbitlang/core/json` has alias `json`, accessed
as `@json`.

A custom alias can be defined in the `import` field.

---

## Importing and Using Packages

### Accessing Imported Entities

Use `@pkg_alias` to access:

```moonbit
let data = @json.parse(input)
```

### `using` Syntax

Import symbols from another package directly:

```moonbit
using @json.{ JsonValue, parse }

// With pub: reexports the symbols
pub using @json.{ JsonValue }
```

---

## Internal Packages

Code in `a/b/c/internal/x/y/z` is only available to packages `a/b/c`
and `a/b/c/**`.

This enforces encapsulation at the filesystem level.

---

## Access Control

### Functions and Variables

- Default: **invisible** to other packages
- `pub fn` / `pub let`: visible to other packages

### Aliases

- Default: follow the visibility of the original definition (for function/method alias)
- Default: invisible (for type alias, `using`)
- `pub` modifier can be added

### Types (struct, enum)

Four visibility levels:

| Modifier    | Name           | Outside can...                                      |
|-------------|----------------|-----------------------------------------------------|
| `priv`      | Private        | Nothing (completely hidden)                         |
| _(default)_ | Abstract       | See the type name only (internal representation hidden) |
| `pub`       | Readonly       | Read fields, pattern match; **cannot** construct or mutate |
| `pub(all)`  | Fully public   | Construct, read, mutate freely                      |

```moonbit
// Abstract (default) — name visible, internals hidden
struct InternalState { data : Array[Int] }

// Readonly — can read fields outside, cannot construct
pub struct Config { name : String; version : Int }

// Fully public — full access
pub(all) struct Point { x : Double; y : Double }

// Private fields in public struct
pub(all) struct User {
  name : String
  priv password_hash : String  // hidden from outside
}
```

**Readonly type example**:

```moonbit
// In package A:
pub struct RO { field : Int }

// In package B:
fn use_ro(r : @a.RO) -> Int {
  r.field           // OK: can read
  // { field: 4 }   // ERROR: cannot construct
  // { ..r, field: 8 } // ERROR: cannot mutate
}
```

**Consistency rule**: A `pub` type, function, or variable cannot be defined
in terms of a private type.

### Traits

| Modifier      | Name          | Outside can...                                  |
|---------------|---------------|-------------------------------------------------|
| `priv trait`  | Private       | Nothing (completely hidden)                     |
| _(default)_   | Abstract      | See trait name; methods not exposed              |
| `pub trait`   | Readonly      | Invoke methods; only current package can implement |
| `pub(open) trait` | Fully public | Invoke methods; anyone can implement          |

Abstract and readonly traits are **sealed**: only the defining package can
implement them.

### Trait Implementations

Implementations have independent visibility:

```moonbit
pub impl Show for MyType with to_string(self) { ... }
// The "pub" makes MyType considered as implementing Show outside this package
```

### Implementation Location Rules

- **Methods**: only the package that defines a type can define methods for it
  (exception: local methods, which are always private)
- **Trait impls**: only the package of the type OR the package of the trait
  can define an implementation

These rules ensure coherence: globally unique `Type: Trait` pairs.

---

## Virtual Packages

> Experimental feature.

Virtual packages serve as interfaces that can be replaced at build time.
Currently only support plain functions.

### Defining

In `moon.pkg.json`:

```json
{ "virtual": { "has-default": true } }
```

Interface file `pkg.mbti`:

```moonbit
package "full-package-name"

fn add(Int, Int) -> Int
```

### Implementing

Default implementation: set `has-default: true` and implement normally.

Third-party implementation: set `implements` to the target package name:

```json
{ "implements": "user/project/virtual-pkg" }
```

### Overriding

```json
{
  "overrides": [
    { "path": "user/project/virtual-pkg", "with": "user/project/impl-pkg" }
  ]
}
```
