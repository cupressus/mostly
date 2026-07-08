---
name: "mkdocs-config"
description: "Use when editing mkdocs.yml for this repository, including MkDocs navigation, MkDocs Material settings, notebook page integration, and documentation structure. Covers concise doc-focused config changes without touching src code."
applyTo: "mkdocs.yml"
---

# MkDocs Configuration Guidelines

## Scope
- Keep changes limited to documentation configuration. Do not expand into source-code changes unless the user explicitly re-scopes it.
- Stop at the configuration boundary. If a request requires build-system or plugin changes outside MkDocs scope, report the gap.

## Repository Defaults (Preserve Unless Justified)
- Keep current MkDocs Material theme, language (`en`), light/dark palette toggle, and Roboto / Roboto Mono fonts. Change only if the task or repository state indicates a specific UX or accessibility need.
- Preserve notebook integration through `mkdocs-jupyter` and `execute: False` unless the task explicitly asks to change execution behavior.
- Preserve existing repository metadata: `repo_name`, `repo_url`, and social links unless the task specifically targets an update.

## Navigation & Structure
- Keep navigation clear and reader-focused.
- Reflect Diataxis principles where practical. Group pages under clear sections: Tutorials, How-to Guides, Reference, Explanation.
- Use concise, stable page titles in `nav`.
- Keep file paths in `nav` accurate and aligned with the docs tree.
- Preserve the current top-level flow unless the task calls for a restructure.
- If Diataxis sections are introduced or expanded, prefer uncommenting or replacing placeholder `nav` entries with real pages rather than inventing speculative structure.

## Content Type Preferences
- Prefer Markdown pages for most docs. Use notebooks only when the page genuinely benefits from executable, stepwise examples.
- When adding tutorial notebooks, place them under `Tutorials` in `nav` and give them concise task- or concept-based titles.

## Change Strategy
- Prefer additive, low-risk config changes over broad restructuring.
- When enabling MkDocs Material features, use only settings that fit the current site structure and content.
