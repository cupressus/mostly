---
name: "mkdocs-config"
description: "Use when editing mkdocs.yml for this repository, including MkDocs navigation, MkDocs Material settings, notebook page integration, and documentation structure. Covers concise doc-focused config changes without touching src code."
applyTo: "mkdocs.yml"
---

# MkDocs Configuration Guidelines

- Keep changes limited to documentation configuration. Do not expand the task into source-code changes unless the user explicitly re-scopes it.
- Preserve the existing MkDocs Material setup unless the task explicitly asks for a theme or feature change.
- Preserve the current Material theme basics unless explicitly asked to change them: `language: en`, the light/dark palette toggle, and the Roboto / Roboto Mono fonts.
- Keep navigation clear and reader-focused.
- Preserve the current top-level flow unless the task calls for a restructure: `Introduction` first, then Diataxis-aligned sections.
- Reflect Diataxis in `nav` where practical. Group pages under clear sections such as Tutorials, How-to Guides, Reference, and Explanation.
- Use concise, stable page titles in `nav`.
- Keep file paths in `nav` accurate and aligned with the docs tree.
- Prefer Markdown pages for most docs. Use notebooks only when the page genuinely benefits from executable, stepwise examples.
- Preserve notebook integration through `mkdocs-jupyter` unless the task explicitly asks to change it.
- Preserve `execute: False` for `mkdocs-jupyter` unless the task explicitly asks to change notebook execution behavior.
- When adding tutorial notebooks, place them under `Tutorials` in `nav` and give them concise task- or concept-based titles.
- Prefer additive, low-risk config changes over broad restructuring.
- When enabling MkDocs Material features, use only settings that fit the current site structure and content.
- Preserve existing repository metadata such as `repo_name`, `repo_url`, and social links unless the task explicitly asks to update them.
- If Diataxis sections are introduced or expanded, prefer uncommenting or replacing placeholder `nav` entries with real pages rather than inventing speculative structure.
