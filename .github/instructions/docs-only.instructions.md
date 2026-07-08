---
name: "docs-only"
description: "Use when writing or revising documentation for this repository, including Diataxis-based docs, MkDocs pages, and README content. Covers concise active-voice writing, documentation-only scope, and MkDocs Material conventions."
applyTo: "docs/**, README.md"
---

# Documentation Guidelines

- Treat each page as exactly one Diataxis type: tutorial, how-to, reference, or explanation.
- Keep each page focused on a single Diataxis type. If content mixes types, split it and cross-link.
- Use reader-friendly language, active voice, and short sentences.
- Prefer brevity over complexity.
- Make task steps concrete and easy to follow.
- Preserve existing project terminology unless the task explicitly asks for a terminology cleanup.
- Follow MkDocs and MkDocs Material conventions where relevant, including clear headings, code fences, admonitions, tabs, and internal cross-links.
- Keep documentation changes within doc-facing files. Do not expand the task into source-code changes unless the user explicitly re-scopes it.