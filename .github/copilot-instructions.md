---
description: "Use when writing Python docstrings or documenting Python APIs."
applyTo: "**/*.py"
---

# Python docstrings

- Use Google-style sections and PEP 257 formatting: a concise summary, followed
  by a blank line before any further detail. Add longer explanation only when needed. Use an imperative summary for functions and methods.
- Document public modules, classes, functions, and methods. Keep obvious
  methods short; add detail for behavior that a caller cannot infer.
- Use `Args`, `Returns` or `Yields`, `Raises`, and `Examples` only when useful.
  For Pydantic models, describe constructor fields under `Attributes` so the
  API renderer can infer their annotations.
- Treat annotations and signatures as the source of truth for types and
  defaults. Explain meaningful default behavior without mechanically repeating
  signatures or listing methods already documented separately.
- Describe the contract: constraints, units, shapes, boundary cases, return
  semantics, side effects, and exceptions that callers need to handle. Match
  the actual implementation, including validation and numerical limitations.
- Where an example clarifies behavior, use a small, deterministic doctest with
  public imports. Run the example or a test rather than guessing its output.
- Keep reference docstrings concise; put extended tutorials and explanations in
  the documentation pages.