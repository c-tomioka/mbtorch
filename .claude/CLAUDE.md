# CLAUDE.md – MbTorch

This repository contains **MbTorch**, a lightweight AI/ML framework written in **MoonBit**, targeting **WebAssembly** and edge environments.  
The main goals are:
- Define and train small-to-medium neural networks in MoonBit
- Import pretrained models via **ONNX** and **safetensors**
- Run inference and lightweight fine-tuning in browsers and on edge devices
- Provide a MoonBit-native model format (e.g. `.mbt`) for efficient save/load

You (Claude) will frequently be used with **plan mode** and **skills**.  
Always read and follow this file **before** making any changes.[3][1]

For general MoonBit project conventions (package layout, test file naming,
tooling commands), see `AGENTS.md`.  
For MbTorch-specific architecture and workflow, this `CLAUDE.md` and
`.claude/skills` take precedence.

***

## 1. Tech stack & scope

- **Language**: MoonBit (WASM-first, small binaries, fast compilation)[4][5]
- **Domain**: Neural networks, autograd, edge/browsers ML, model I/O (ONNX, safetensors, `.mbt`)[6][7]
- **Non-goals** (for now):
  - Massive distributed training
  - Full ONNX operator coverage
  - GPU-specific backends

Focus on:
- Correctness first (math, gradients, serialization)
- Clear architecture and responsibilities between modules
- Test-driven development (TDD) for all new work[8][9]

***

## 2. Repository structure (high-level)

Treat this as the “map” of the codebase. Do **not** add new top-level folders without an explicit design decision.

```text
mbtorch/
  core/
    tensor/      # Tensor types and forward-only numeric ops
    autograd/    # Autograd engine and gradient definitions
  nn/            # Layers and high-level neural network building blocks
  optim/         # Optimization algorithms (SGD, Adam, etc.)
  io/            # Model import/export (ONNX, safetensors, .mbt)
  examples/      # Demos, browser/edge examples, small end-to-end scripts
  tests/         # Unit and integration tests
  .claude/       # CLAUDE.md (this file) and SKILLs for Claude Code
```

**Layering rules (very important):**

- `core`  
  - Provides tensors, numerical ops, and autograd.  
  - **Must not** depend on `nn`, `optim`, or `io`.
- `nn`  
  - Depends on `core` only.
- `optim`  
  - Depends on `core` and `nn`.
- `io`  
  - May depend on `core` and `nn`, but `core` must never depend back.
- `examples`, `tests`  
  - Can depend on everything, but should avoid re-implementing library logic.

If you propose code that violates these layering rules, stop and suggest a refactor instead of introducing a bad dependency.[10][11]

***

## 3. Coding and design principles

When you generate or edit code, follow these principles:

1. **TDD first**
   - Write or update tests **before** making non-trivial changes.
   - Every new public API must have tests.
   - Bug fixes should come with regression tests.[12][8]

2. **Small, composable modules**
   - Prefer splitting large files into focused modules (e.g. `ops_basic`, `ops_nn`).
   - Keep `core/tensor` and `core/autograd` conceptually separate, even if re-exported together.

3. **Public API consistency**
   - Keep naming and signatures consistent (`Tensor`, `Linear`, `Model.from_onnx`, etc.).
   - Avoid surprising behavior; prefer explicit, predictable APIs.
   - Before finalizing a new API, write a short usage snippet in a comment or example.

4. **Safety over cleverness**
   - Prefer clear code over micro-optimizations.
   - Avoid hidden global state.
   - Be explicit about shapes, dtypes, and error conditions in critical paths.

***

## 4. Testing & commands

Assume the standard MoonBit toolchain is installed.

Typical commands (do **not** hardcode exact flags unless they exist in the repo):

- Run all tests:

```bash
moon test
```

- Build for WASM (browser/edge):

```bash
moon build --target wasm32-unknown-unknown
```

When adding tests:

- Unit tests belong close to the module they test (e.g. `tests/core_tensor_*.mbt`).  
- Integration tests (e.g. “training reduces loss”, “ONNX import parity”) go in dedicated files under `tests/`.  
- Prefer fast tests; if a test is slow, mark it clearly.

If you are unsure how tests are structured, first scan `tests/` and follow existing patterns.[13][1]

***

## 5. How to work with this repo (Claude workflow)

Before writing or modifying code:

1. **Plan first**  
   - Use plan mode for any task involving more than ~2 files.  
   - Outline which files you will read, what you will change, and what tests you will add or update.[14][3]

2. **Read the relevant files**  
   - For `core` changes: read existing tensor / autograd code first.  
   - For `io` changes: read existing ONNX / safetensors utilities and any `.mbt` code.
   - For public API changes: search for usages in `examples/` and `tests/`.

3. **Apply TDD**  
   - Write or update tests to capture the intended behavior.  
   - Only after that, implement the minimal changes to make tests pass.  
   - Then refactor, keeping tests green.

4. **Minimize blast radius**  
   - Avoid large unrelated refactors in the same change as functional changes.  
   - If you see structural issues, propose a separate refactor step.

5. **Document as you go**  
   - If you add or change public APIs, update README, `docs/`, or `examples/` as needed.  
   - Keep documentation concise and practical.

***

## 6. Interaction with `.claude/skills`

This repository uses **skills** to give you more detailed instructions for specific tasks.[15][16]

- The root `CLAUDE.md` (this file) defines **global rules**:
  - Architecture, directory layout, workflows, basic commands.
- `.claude/skills/*/SKILL.md` files provide **task-specific behavior**, for example:
  - MbTorch architecture understanding
  - MbTorch TDD rules
  - MoonBit language evaluation
  - MoonBit error investigation
  - MbTorch I/O (ONNX / safetensors / `.mbt`) guidelines

When working on a task:
- Use the relevant skill(s) if available, **in addition to** this file.
- Do not contradict rules defined here.  
- If there is a conflict, assume this `CLAUDE.md` takes precedence and point out the inconsistency.

***

## 7. Things you must NOT do

- Do not introduce dependencies from `core` to `io`, `nn`, or `examples`.  
- Do not add new public APIs without tests and at least one usage example or doc snippet.  
- Do not silently change the behavior of existing public APIs; propose migrations or versioning if needed.  
- Do not remove or rewrite tests without a clear justification.

If a user asks you to do something that conflicts with these rules, explain the trade-offs and propose a safer alternative.

***

If you need more information that is not in this file, ask the user for clarification rather than guessing.
