---
name: paper-summary
description: Summarize an arXiv paper into this repository's paper-reading markdown format, enrich citations with a2b from conda base or manually when a2b is unavailable, preserve core formulas, and save useful figures locally
---

## What I do

This skill turns an arXiv paper link in a markdown note into a repository-native paper summary.

I follow the local paper-reading style already used in this repository, especially the patterns under `AI/ml/DL/CV/Paper-Reading/` and the reading workflow described in `research/Research-Guide/03-Pipeline.md`.

I write concise Chinese-dominant mixed-language summaries in Typora-friendly markdown, place core math formulas under the relevant `Core Mechanism` or `Pipeline` points, and store useful figures under the local `assets/<note-file-base>.assets/` directory.

## Use this skill when

- A markdown note contains an arXiv link and needs a structured paper summary
- You want the summary in this order: Takeaway, Motivation, Core Mechanism, Pipeline, Pros, Cons
- You want key formulas shown directly under the relevant mechanism bullets
- You want useful figures actually saved locally next to the note
- You want citation metadata enriched with `a2b`, or manually injected when `a2b` is unavailable
- You require every summarized paper entry to include a citation count field

## Local repository conventions to follow

- Prefer the style under `AI/ml/DL/CV/Paper-Reading/`
- Prefer the citation style shown in `AI/ml/DL/CV/Paper-Reading/AGENTS.md`
- Keep summaries compact and useful for later review
- Match nearby files before inventing a new structure
- Write explanations mainly in Chinese, but keep technical nouns, module names, benchmark names, and short technical phrases in English when that is clearer
- Keep Markdown readable in Typora
- Preserve existing frontmatter, title, TOC, and unrelated note content unless the task explicitly asks to restructure them

## Required note structure

Write the summary in Chinese-dominant mixed-language style.

Nearby-file structure is the primary rule, but the default language baseline remains Chinese-dominant mixed-language unless the surrounding directory already has a strong and stable alternative convention.
Use this section order as the default pattern, but adapt to the surrounding file when it already uses a stable local variation such as `Prior` or `Background` instead of `Motivation`.

By default, place formulas and figures inline under the relevant `Core Mechanism` points instead of collecting them into detached summary sections.

## Takeaway
- One or two sentences only
- State the paper's main claim, setting, and why it matters

## Motivation
- What problem does the paper solve?
- What gap or limitation in prior work motivates it?
- Why does the problem matter?

`Prior` or `Background` is also acceptable when that matches the local note style better.

## Core Mechanism
- Start `Core Mechanism` with one architecture figure when the paper has a clear model structure and the figure is genuinely useful for orientation
- After the first figure, explain each major innovation point using `What`, `Why`, and `How`
- Only create mechanism bullets for genuinely important innovations, not trivial background details

### What
- Explain what the innovation is
- Include the core formula or mathematical relation when needed
- Explain the math essence directly here instead of splitting it into a separate subsection

### Why
- Explain what problem the innovation addresses
- Explain what weakness in prior work motivates it

### How
- Explain where the innovation sits in the model or algorithm
- Explain what inputs and outputs it has and how it interacts with other modules
- Explain how it is used during training and/or inference when the paper makes that clear

## Pipeline
1. Input / setup
2. Main processing stages
3. Training or inference flow
4. Output or evaluation flow

Pipeline should stay concise and operational.

- Do not force Python code blocks in `Pipeline`
- Focus on the end-to-end process rather than repeating the detailed innovation analysis from `Core Mechanism`
- Keep the section step-based and useful for later review

## Pros
- Concrete strengths only

## Cons
- Concrete limitations only

## Formula and figure placement
- Place formulas directly under the mechanism point they explain
- Place figures directly under the explanation they support
- Use a separate `Math Formula` or `Figure Notes` section only when the nearby file already prefers that structure
- Do not create detached formula dumps or image galleries

## Citation style

Prefer this citation format for new entries:

```markdown
- **Paper Title**. Author1 Author2 et.al. **Venue**, **Year**, ([link](URL)).
```

Always append a citation count field for every summarized paper entry. The preferred local style is:

```markdown
[(Arxiv)](URL) [(S2)](URL) [(Code)](URL) (Citations __N__)
```

`(Citations __N__)` is mandatory, not optional. If the citation count comes from Semantic Scholar, include the `[(S2)](...)` link when available. If Semantic Scholar is rate-limited or unavailable, use a reasonable fallback such as arxiv.gg, Google Scholar snippets, OpenAlex, Crossref, DBLP-connected pages, publisher pages, or another stable bibliographic/search source, then still write `(Citations __N__)`. Do not fabricate an `[(S2)](...)` link when the count did not come from Semantic Scholar.

Only if no citation count can be verified after reasonable fallback attempts, write `(Citations __unknown__)` in the entry and mention the failed sources in the final report. Do not omit the citation field.

## a2b and citation enrichment

Use `a2b` only for bibliography enrichment, not for full paper extraction.

Always try `a2b` from the conda `base` environment first. Prefer `conda run` because it does not depend on shell activation state:

```bash
conda run -n base a2b --arxiv https://arxiv.org/abs/<paper-id>
conda run -n base a2b path/to/file.md
```

If `conda run -n base a2b ...` fails because `a2b` is not installed or the base environment is unavailable, do not stop at "a2b unavailable" and do not leave the citation incomplete. Manually gather and inject citation metadata from primary or stable bibliographic sources:

1. arXiv abstract page for title, authors, arXiv ID, submitted/revised dates, DOI, and official code links when present.
2. Semantic Scholar for S2 URL and citation count when reachable.
3. Fallback citation-count sources such as arxiv.gg, OpenAlex, Google Scholar snippets, Crossref, publisher pages, or reputable indexed paper pages when Semantic Scholar is unavailable.
4. DBLP, OpenReview, CVF, ACL Anthology, NeurIPS proceedings, ICLR/ICML/OpenReview pages, or the official project page for venue/year/code when more authoritative than arXiv metadata.

Manual citation enrichment should update the note entry in the local style, for example:

```markdown
[(Arxiv)](URL) [(S2)](URL) [(Code)](URL) (Citations __N__)
```

For ordinary metadata fields, if a field cannot be verified after trying reasonable sources, omit that field rather than guessing; mention the gap briefly in the final report. Citation count is the exception: always include `(Citations __N__)` when verified, or `(Citations __unknown__)` if it truly cannot be verified.

Important limitation:

- `a2b` does not extract the paper body
- `a2b` does not download figures
- `a2b` does not preserve formulas from the paper
- `a2b` only produces or injects bibliography metadata

Use `a2b` or the manual fallback to improve the citation entry inside the markdown note, but do not treat either as the source of the summary.

## Source priority and fallback workflow

Use sources in this priority order:

1. The paper itself:
   - arXiv abstract page
   - arXiv HTML or ar5iv-style HTML
   - arXiv source files
   - PDF
2. Paper-adjacent primary materials:
   - official project page
   - official code repository
   - official appendix or supplementary material
3. High-quality auxiliary sources:
   - Semantic Scholar
   - author or lab posts
   - official project blogs
   - reputable technical explainers or videos
4. Everything else

The paper remains the sole ground truth for:

- contribution claims
- objective and method
- equations
- datasets and metrics
- reported results
- limitations and conclusions

Auxiliary sources may help interpretation, but must never override the paper.

## Allowed use of Semantic Scholar

Semantic Scholar is allowed as an auxiliary research tool.

Use it for:

- verifying title, authors, venue, year, and identifiers
- checking citation count as a rough context signal
- finding references, citations, and related papers
- quick relevance triage through metadata
- discovering neighboring work worth linking

Do not use Semantic Scholar as the primary basis for:

- the final summary
- the method explanation
- equations
- exact result claims
- limitations
- nuanced comparisons

Treat these Semantic Scholar signals as secondary only:

- TLDR
- auto-generated summaries
- inferred topics
- fields of study
- ranking or recommendation signals
- citation intent labels

If the note mentions Semantic Scholar-derived information, label it explicitly, for example:

- Semantic Scholar lists ...
- Semantic Scholar shows a citation count of ...
- Semantic Scholar recommends ...

## Allowed use of blogs and explainers

Blogs, videos, and explainer posts are allowed only after reading the paper.

Use them only to:

- improve intuition
- clarify a difficult mechanism
- explain a figure more accessibly
- infer likely implementation interpretation when the paper is terse
- collect better terminology for describing the method clearly

Prefer these auxiliary sources:

- posts by the paper authors
- the authors' lab or project site
- official project blogs
- reputable research organizations
- reputable engineering organizations
- strong technical explainers with named authors and references

Reject these sources:

- anonymous posts
- SEO-style content farms
- ad-driven summary pages
- sources with no author or no date
- sources with no traceable link to the paper
- sources whose claims cannot be checked against the paper

If a blog adds interpretation that is not explicit in the paper, label it as external context or intuition, not as part of the paper's stated claims.

## Strict workflow for selecting figures and formulas

Follow this workflow in order. Do not skip steps.

### Step 1: Identify the paper source

Use the arXiv abstract page as the entry point.

If the arXiv page links to source files, HTML rendering, or the PDF, prefer sources in this order:

1. arXiv HTML or ar5iv-style HTML if it preserves section structure and formulas clearly
2. arXiv source files when figure names and LaTeX equations are needed
3. PDF only when the HTML or source path is insufficient

Do not rely on third-party summaries as the primary source.

### Step 2: Read before extracting

Before copying anything, read enough of the paper to identify:

- the main claim
- the central mechanism
- the minimal end-to-end pipeline
- the one or two equations without which the method is hard to understand
- the one to three figures that explain the method better than text alone

Do not extract formulas or figures before understanding their role.

### Step 3: Figure selection and fetching rules

Only keep figures that satisfy at least one of these purposes:

- explain the overall architecture
- show the method pipeline
- clarify a key module or algorithm step
- summarize the main quantitative result when the result is central to the claim

Prefer figures in this order:

1. architecture diagram when the paper has a clear model structure
2. pipeline or training/inference flow diagram
3. key module illustration
4. one main results figure only if no mechanism-supporting figure is stronger

- Guardrail: use the architecture-first rule only when the architecture figure genuinely helps orient the reader.

Do not keep:

- teaser images
- qualitative examples without clear explanatory value
- redundant ablation plots
- repeated tables or near-duplicate diagrams
- decorative or promotional figures

Hard limits:

- default maximum: 3 figures per paper note
- if one figure already explains the method clearly, stop there
- only exceed 3 if the user explicitly asks for a more detailed note

Default fetching rule:

- If the paper contains a clear architecture diagram, pipeline figure, or single high-value result figure, fetch and save at least one such figure locally instead of only mentioning it in prose.
- Only skip figure fetching when the paper truly has no high-value figure, the figure is unreadable from available sources, or the user explicitly asks for a text-only summary.
- If you skip fetching, say briefly why.

### Step 4: Formula selection rules

Only keep formulas that are essential to understanding the paper.

Eligible formulas usually include:

- the core objective function
- the main update rule
- the scoring, matching, attention, or energy function
- the final loss if it defines the method
- one theorem statement only if the paper is primarily theoretical

Do not include:

- routine background formulas from prior work unless reused critically
- every intermediate derivation
- long proof steps
- notation-heavy expansions that do not change understanding
- appendix-only equations unless they are central

Hard limits:

- default maximum: 2 block formulas
- allow 3 only if the method truly has multiple irreducible objectives or stages
- prefer one core formula plus one auxiliary formula over a long chain of equations

### Step 5: Extraction quality rules

When copying a figure:

- save it locally under `assets/<note-file-base>.assets/`
- use a stable descriptive filename when possible
- embed it with a relative markdown path
- add a one-line explanation if the meaning is not obvious
- prefer actually embedding the saved figure near the relevant mechanism point instead of leaving only a recommendation note

When copying a formula:

- preserve LaTeX faithfully
- use block math for important equations: `$$...$$`
- use inline math only for short symbols in prose
- after each important formula, add a short explanation of what it means
- define only the non-obvious symbols, not every symbol mechanically

### Step 6: Relevance test before keeping an item

For each candidate figure or formula, ask:

- Does this help explain the takeaway?
- Does this clarify the core mechanism or pipeline?
- Would removing it make the note meaningfully worse?

If the answer is no to all three, do not include it.

### Step 7: Summarization-first rule

Write the summary first.
Then add only the figures and formulas still needed after the prose is complete.

The note must remain useful even if all figures are removed.
Figures and formulas are supporting evidence, not the backbone of the note.
But if a single figure materially improves understanding, include it rather than merely describing it.

### Step 8: Repository-style constraints

Match local repository style:

- keep the note compact
- prefer explanation over transcription
- do not turn the note into a full paper dump
- preserve Typora-friendly markdown
- keep local assets near the note
- follow nearby `Paper-Reading` examples before inventing a new layout

## PDF-only fallback workflow

If HTML or source files are unavailable or poor, use this strict PDF-first fallback:

1. Read in this order:
   - title
   - abstract
   - introduction
   - conclusion or discussion
2. Write a provisional summary before deeper extraction:
   - problem
   - contribution
   - main result
   - stated limitation
3. Scan all figures and tables before reading methods in detail.
4. Keep only claim-bearing visuals:
   - one overall method or pipeline figure if needed
   - one main result figure or table
   - one key comparison or ablation only if it changes the conclusion
5. Read only the body paragraphs needed to explain the kept visuals.
6. Extract equations only after the core claims and kept visuals are already known.
7. Keep only equations that are essential:
   - one core objective or definition
   - one update, inference, or scoring equation if necessary
   - one nonstandard metric or constraint only if results depend on it
8. For every important claim, record:
   - claim
   - evidence location
   - limitation or boundary
9. Paraphrase by default:
   - never paste the full abstract
   - never dump long captions
   - never copy long derivations
   - never copy proof-heavy blocks into the note
10. If PDF parsing quality is poor, keep only high-confidence claims supported by multiple parts of the paper.

## Figure and formula checklist

Before finishing, verify:

- no more than 3 figures unless explicitly needed
- no more than 2 core block formulas by default
- every kept figure has a clear explanatory purpose
- every kept formula supports the mechanism directly
- all assets are saved locally with relative paths
- formulas sit under the relevant mechanism or pipeline points by default
- at least one high-value figure is fetched and embedded when the paper clearly has one and the source is usable
- the prose summary still stands on its own without the raw paper

## Required asset placement

When saving figures for a note:

- put them under `assets/<note-file-base>.assets/`
- reference them with a relative markdown path matching nearby file style, such as `assets/<note-file-base>.assets/<image-file>` or `./assets/<note-file-base>.assets/<image-file>`

Match the repository's existing local asset pattern.

## Must do

1. Read the target markdown note and extract the arXiv link.
2. Inspect nearby notes in the same directory and follow their local style.
3. Try citation enrichment with `conda run -n base a2b ...`; if `a2b` is unavailable, manually gather citation metadata and inject it into the note entry.
4. Verify and write a citation count for every summarized paper entry as `(Citations __N__)`; if no count can be verified after reasonable fallbacks, write `(Citations __unknown__)` and report the attempted sources.
5. Summarize the paper in Chinese-dominant mixed-language style using the default section order, but adapt to stable nearby conventions when needed.
6. Preserve the paper's most important formulas in markdown math directly under the relevant mechanism or pipeline points.
7. Save and embed at least one high-value figure locally into the matching `assets/<note>.assets/` folder when the paper clearly contains one and the source is usable.
8. Keep the final note readable in Typora.
9. Keep the final summary grounded in the paper even if auxiliary sources are consulted.

## Must not do

- Do not claim that `a2b` extracts paper content
- Do not dump the whole abstract or whole paper into the note
- Do not include formulas that are not central to the method
- Do not move key formulas into a detached section when they belong under a specific mechanism point
- Do not include decorative or redundant figures
- Do not stop at a prose-only `Figure Note` when a high-value figure is available and can be saved locally
- Do not let Semantic Scholar or a blog override the paper
- Do not hotlink random images if a stable local copy can be saved
- Do not break existing relative paths or nearby note conventions
- Do not rewrite or remove existing frontmatter, title, TOC, or unrelated note scaffolding without a task-specific reason
- Do not overwrite unrelated note content

## Final grounding rule

The final note must still stand on its own if all auxiliary sources are removed.

Auxiliary sources may improve understanding, but the final summary must remain grounded in the paper itself.
