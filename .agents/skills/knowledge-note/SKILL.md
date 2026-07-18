---
name: knowledge-note
description: Format any learning topic into a structured knowledge note following the repository's note-taking conventions (Takeaway → Motivation → Core Mechanism → Pipeline → Pros → Cons), with Chinese-dominant mixed-language prose and Typora-friendly markdown.
---

## What I do

This skill turns raw learning material — concepts, algorithms, theorems, frameworks, techniques, or any knowledge point — into a structured, review-friendly markdown note following this repository's house style.

I follow the same section architecture as the `paper-summary` skill (`Takeaway → Motivation → Core Mechanism → Pipeline → Pros → Cons`) but generalized for any knowledge domain, not just academic papers. The format is already used extensively under `AI/ml/CV/Paper-Reading/` and `AI/ml/DL/`.

I write concise Chinese-dominant mixed-language summaries in Typora-friendly markdown, embed core formulas under the relevant mechanism or pipeline points, and store useful figures under the local `assets/<note-file-base>.assets/` directory.

## Use this skill when

- A markdown note needs a structured summary for a knowledge point (concept, algorithm, theorem, framework, technique)
- You want the summary in this order: Takeaway, Motivation (optional), Core Mechanism, Pipeline (optional), Pros (optional), Cons (optional)
- You want key formulas shown directly under the relevant mechanism or pipeline bullets
- You want useful figures stored locally next to the note
- You are creating or reformatting a note in any subject area (math, AI, engineering, etc.)

## Local repository conventions to follow

- Prefer the style under `AI/ml/CV/Paper-Reading/` and `AI/ml/DL/` as primary references
- Keep summaries compact and useful for later review
- Match nearby files before inventing a new structure
- Write explanations mainly in Chinese, but keep technical nouns, module names, benchmark names, and short technical phrases in English when that is clearer
- Keep Markdown readable in Typora
- Preserve existing frontmatter, title, TOC, and unrelated note content unless the task explicitly asks to restructure them
- Use `---` horizontal rules to separate distinct knowledge entries within the same note

## Required note structure

Write the summary in Chinese-dominant mixed-language style.

Nearby-file structure is the primary rule, but the default language baseline remains Chinese-dominant mixed-language unless the surrounding directory already has a strong and stable alternative convention.

Use this section order as the default pattern. Sections marked **(optional)** should only be included when the content genuinely adds value; omit them otherwise.

### Entry format (per knowledge point)

Each knowledge point within a note should follow this structure:

```markdown
- **Concept/Algorithm/Theorem Name**. Attribution info when applicable.

  - Takeaway: One or two sentences capturing the core idea and why it matters.

  - Motivation: (optional) What problem does this solve? What gap does it fill?

  - Core Mechanism: Detailed explanation with formulas and figures inline.

  - Pipeline: (optional) Step-by-step algorithmic flow.

  - Pros: (optional) Concrete strengths.

  - Cons: (optional) Concrete limitations.
```

---

## Takeaway

- One or two sentences only
- State the core claim, the setting, and why it matters
- The reader should understand the essence without reading further

## Motivation (optional)

- What problem does this knowledge point solve?
- What gap or limitation in prior approaches motivates it?
- Why does the problem matter?
- Omit this section entirely if the motivation is obvious or the note is purely definitional

## Core Mechanism

- Start with one architecture or concept figure when available and genuinely useful for orientation
- After the first figure, explain each major aspect using **What**, **Why**, and **How**
- Only create mechanism bullets for genuinely important innovations or insights, not trivial background details
- For theoretical notes (math, physics, etc.), the Core Mechanism is where definitions, theorems, and key derivations live

### What
- Explain what the concept/mechanism/innovation is
- Include the core formula or mathematical relation when needed
- Explain the math essence directly here instead of splitting it into a separate subsection

### Why
- Explain what problem this addresses
- Explain what weakness in prior approaches motivates it

### How
- Explain where this sits in the broader system or theory
- Explain inputs, outputs, and interactions with other components
- For algorithms: explain training and/or inference usage when applicable
- For theorems: explain conditions, guarantees, and proof intuition

## Pipeline (optional)

Only include Pipeline for knowledge points where the end-to-end flow is important for understanding — typically algorithms, protocols, or multi-stage processes.

1. Input / setup
2. Main processing stages
3. Training or inference flow (for ML algorithms)
4. Output or evaluation flow

- Keep Pipeline concise and operational
- Do not force Python code blocks unless truly helpful
- Focus on the end-to-end process rather than repeating the detailed analysis from Core Mechanism
- Keep the section step-based and useful for later review

## Pros (optional)

- Concrete strengths only
- No vague praise — each pro should be verifiable

## Cons (optional)

- Concrete limitations only
- Include known failure modes, assumptions that may not hold, efficiency bottlenecks, etc.

## Formula and figure placement

- Place formulas directly under the mechanism point they explain
- Place figures directly under the explanation they support
- Use a separate `Math Formula` or `Figure Notes` section only when the nearby file already prefers that structure
- Do not create detached formula dumps or image galleries

### Formula style
- Preserve LaTeX faithfully
- Use block math for important equations: `$$...$$`
- Use inline math only for short symbols in prose: `$...$`
- After each important formula, add a short explanation of what it means
- Define only the non-obvious symbols, not every symbol mechanically

### Figure style
- Save figures locally under `assets/<note-file-base>.assets/`
- Use a stable descriptive filename
- Embed with a relative markdown path: `![description](assets/<note-file-base>.assets/<image-file>)`
- Add a one-line explanation if the meaning is not obvious

## Entry separation

When a note contains multiple knowledge entries, separate them with a `---` horizontal rule:

```markdown
- **Entry A**. ...

  - Takeaway: ...

---

- **Entry B**. ...

  - Takeaway: ...
```

## Multi-level notes

For hierarchical notes (e.g., a topic with sub-topics), use headings to organize:

```markdown
## Broad Topic

### Sub-topic 1

- **Knowledge point 1**. ...

  - Takeaway: ...

- **Knowledge point 2**. ...

  - Takeaway: ...

### Sub-topic 2

- **Knowledge point 3**. ...

  - Takeaway: ...
```

## Must do

1. Identify the knowledge point's domain and match the local directory conventions.
2. Inspect nearby notes in the same directory and follow their local style.
3. Write the summary in Chinese-dominant mixed-language style using the section order: Takeaway → Motivation (if applicable) → Core Mechanism → Pipeline (if applicable) → Pros (if applicable) → Cons (if applicable).
4. Place core formulas directly under the relevant mechanism or pipeline points — never in a detached section.
5. Save and embed useful figures locally into `assets/<note-file-base>.assets/` when available.
6. Keep the final note readable in Typora.
7. Separate multiple entries with `---` horizontal rules.
8. Preserve existing frontmatter, title, TOC, and unrelated note content.

## Must not do

- Do not dump raw material verbatim — synthesize and explain
- Do not include formulas that are not central to understanding
- Do not move key formulas into a detached section when they belong under a specific mechanism point
- Do not include decorative or redundant figures
- Do not hotlink external images if a stable local copy can be saved
- Do not break existing relative paths or nearby note conventions
- Do not rewrite or remove existing frontmatter, title, TOC, or unrelated note scaffolding without a task-specific reason
- Do not overwrite unrelated note content
- Do not force sections marked optional when the content doesn't justify them
- Do not use English-only prose unless the surrounding directory already uses that convention — default to Chinese-dominant mixed-language

## Final grounding rule

The final note must stand on its own. Every claim about a knowledge point should be accurate and verifiable. Auxiliary sources may improve clarity, but the core explanation must be self-contained.
