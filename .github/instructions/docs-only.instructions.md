---
name: "docs-only"
description: "Use when writing or revising documentation for this repository, including Diataxis-based docs, MkDocs pages, and README content. Covers concise active-voice writing, documentation-only scope, and MkDocs Material conventions."
applyTo: "docs/**, README.md"
---

# Documentation Guidelines

## Scope
- Keep documentation changes within doc-facing files. Do not expand into source-code changes unless the user explicitly re-scopes it.
- Stop at the documentation boundary. If a request requires code changes, report the exact gap clearly.

## Diataxis Organization
- Prefer each page to have one primary Diataxis type: tutorial, how-to, reference, or explanation.
- Intentionally mixed pages (README, index, release notes, migration guides) are acceptable when splitting would hurt usability. State the primary type and why mixing serves the user.
- If a page mixes types unnecessarily, suggest splitting and cross-linking.

## Writing Style
- Use reader-friendly language, active voice, and short sentences.
- Prefer brevity over complexity.
- Make task steps concrete and easy to follow.
- Preserve existing project terminology unless the task explicitly asks for a terminology cleanup.

## Format & Tooling
- Follow MkDocs and MkDocs Material conventions: clear headings, code fences, admonitions, tabs, internal cross-links.

## Review Mode
- When asked to review or improve docs, prioritize: broken links, stale claims, missing prerequisites, Diataxis mismatches that affect usability, and navigation gaps.
- Avoid large rewrites unless the user asks for edits. Report findings and ask before major restructures.