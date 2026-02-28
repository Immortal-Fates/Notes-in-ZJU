# Agent Guidelines for Paper-Reading Repository

This repository contains documentation notes on computer vision and object detection papers.
**No build, lint, or test commands** - this is a markdown-only documentation repository.

## File Structure

### Naming Convention
- Markdown files: `XX-Title.md` (two-digit prefix + kebab-case title)
- Asset directories: `XX-Title.assets/` (matches markdown file name exactly)

### Organization
- Root level: All `.md` files (e.g., `00-OD-Trends.md`, `01-Basic-Model-Zoo.md`)
- `assets/` directory: Contains subdirectories for each markdown file's assets

## Documentation Style

### Paper Citation Format
```markdown
- **Paper Title**. Author1 Author2 et.al. **Journal/Conference**, **Year**, ([link](URL)).
```

**Examples:**
- `**Attention Is All You Need**. Ashish Vaswani et.al. **arxiv**, **2017**, ([link](https://arxiv.org/abs/1706.03762v7)).`
- `__Focal Loss for Dense Object Detection.__ *Tsung-Yi Lin et al.* __IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017__`

### Content Structure
1. **H1 Header**: Title matching filename (without prefix)
2. **Focus line**: Brief description of document scope
3. **TOC**: `[TOC]` tag for table of contents (if needed)
4. **H2/H3 Sections**: Organize with clear section headers
5. **Paper entries**: Bullet-point style with nested content

### Paper Entry Pattern
```markdown
- **Paper Citation**.

  - Takeaway: One-sentence summary of key contribution
  - Motivation: Background/motivation (optional)
  - Core Mechanism: Detailed explanation
  - Pros: Advantages (bullet list)
  - Cons: Limitations (bullet list)
```

### Nested Content
- Use 2-space indentation for sub-points
- Use 4-space indentation for sub-sub-points
- Keep nested lists clean and consistent

## Markdown Syntax

### Math Formulas
- Display math: `$$ formula $$`
- Inline math: `$ formula $`
- Use proper LaTeX formatting (e.g., `\frac{a}{b}`, `\sum`, `\int`)

### Images
- Reference: `./assets/XX-Title.assets/filename.png`
- Format: `![alt text](./assets/XX-Title.assets/filename.png)`
- With styling: `<img src="./assets/XX-Title.assets/filename.png" alt="text" style="zoom:50%;" />`

### Callout Blocks
```markdown
> [!TIP]
> Your tip content

> [!NOTE]
> Your note content
```

### Links
- Internal: `[text](./other-file.md)` or `(#section-id)`
- External: `[text](URL)`
- Paper links: Include DOI/arxiv links with `[link]` anchor text

## Language and Tone

### Multi-Language Support
- Mix of Chinese and English content is acceptable
- Use Chinese for explanatory notes when natural
- Use English for paper titles, citations, and technical terms
- Maintain consistency within sections

### Writing Style
- **Concise summaries**: Focus on key takeaways
- **Technical depth**: Include formulas and detailed mechanisms
- **Comparative analysis**: Use Pros/Cons lists
- **Prior context**: Explain "why" before "what"

## Asset Management

### Image Files
- Store in corresponding `XX-Title.assets/` directory
- Use descriptive names when possible (e.g., `encoder-decoder-attention.png`)
- Timestamped names are acceptable if no better description (e.g., `image-20251119220845255.png`)

### Asset Links
- Always use relative paths from markdown file
- Format: `./assets/XX-Title.assets/filename.ext`

## Consistency Rules

### Citation Style
- **English papers**: Bold title, period after authors, journal in bold, year in bold
- **Mixed formats**: Both `**Paper Title**. Authors.` and `__Paper Title.__ *Authors*` exist - prefer bold format for new entries

### Section Ordering (per paper entry)
1. Takeaway (if applicable)
2. Prior / Background (if applicable)
3. Core Mechanism (main content)
4. Pros / Cons (if applicable)

### Math Notation
- Use LaTeX math mode consistently
- Variables: $x$, $y$, $\theta$, etc.
- Functions: $\mathrm{SmoothL1}(x)$, $\mathrm{CE}(p_t)$, etc.
- Greek letters: $\alpha$, $\beta$, $\gamma$, etc.
- Subscripts: $p_t$, $t_i^{gt}$, etc.

## Special Patterns

### Model Zoo Entries
- Group by category (e.g., `## MobileNet Zoo`, `## Attention Zoo`)
- Focus on architectural insights and trade-offs
- Include computational complexity when relevant (FLOPs, parameters)

### Loss Function Entries
- Define the loss mathematically
- Explain why it's better/worse than alternatives
- Include parameter choices and typical values
- Note implementation considerations

### Trend Analysis
- Focus on direction and implications
- Connect related papers chronologically
- Highlight key turning points in research
