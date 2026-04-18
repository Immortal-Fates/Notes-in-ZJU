# CLIFF Formula Images Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small set of externally sourced explanatory images to the CLIFF section in `05-Kpts-OD.md` so the two core formulas are visually motivated and easier to review later.

**Architecture:** Use the approved design to source images from the CLIFF paper and official repository first, then add at most one or two clearly relevant third-party explainer images if they improve geometric intuition. Save every chosen image under the note’s local asset folder and place each image directly below the formula it explains, with a short Chinese explanation tying it back to the symbols and coordinate transform.

**Tech Stack:** Markdown, local image assets, web sources (paper/repo/blog), repository note conventions

---

## File Structure

- Modify: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/DL/CV/Paper-Reading/05-Kpts-OD.md`
  - Responsibility: embed the selected explanatory images under the two CLIFF formula blocks and add one-line Chinese explanations.
- Create: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/DL/CV/Paper-Reading/assets/05-Kpts-OD.assets/<selected-image-files>`
  - Responsibility: local copies of official or auxiliary explanatory figures.
- Reference: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/DL/CV/Paper-Reading/docs/superpowers/specs/2026-04-17-cliff-formula-images-design.md`
  - Responsibility: approved design and selection rules.

### Task 1: Collect Candidate Images

**Files:**
- Read: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/DL/CV/Paper-Reading/05-Kpts-OD.md`
- Read: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/DL/CV/Paper-Reading/docs/superpowers/specs/2026-04-17-cliff-formula-images-design.md`
- Create: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/DL/CV/Paper-Reading/assets/05-Kpts-OD.assets/<candidate-files>`

- [ ] **Step 1: Re-read the target formula blocks and lock the two explanation targets**

Read the two formula areas in `05-Kpts-OD.md` and record the two exact image groups needed:

1. `I_{bbox} = [c_x / f_{CLIFF}, c_y / f_{CLIFF}, b / f_{CLIFF}]`
2. crop-to-full translation conversion for `t_X^{full}, t_Y^{full}, t_Z^{full}`

Expected outcome: each later image choice can be justified against one of these two targets.

- [ ] **Step 2: Search official paper/repo sources first**

Search for these official figures before looking anywhere else:

- CLIFF `motivation` figure
- CLIFF `overview` figure
- CLIFF `adjustment` figure

Use the paper source / official repo to identify which figure best explains:

- full image vs crop relationship
- bbox center / bbox size meaning
- `M_crop` vs `M_full` camera relation
- reprojection into the full frame

Expected outcome: 2-4 candidate official images shortlisted.

- [ ] **Step 3: Search auxiliary explainers only if official images leave a gap**

If the official figures do not clearly explain one of the two formula groups, search for at most 1-2 auxiliary explainer images that clarify:

- perspective projection with focal length and image-plane offsets
- crop coordinate system vs full-frame coordinate system

Reject any image that is only qualitative results, contains unreadable labels, or does not map clearly to the symbols in the note.

Expected outcome: at most 1-2 supplemental candidates with clear source attribution.

- [ ] **Step 4: Save only the strongest candidates locally**

Copy or download the shortlisted images into:

`/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/DL/CV/Paper-Reading/assets/05-Kpts-OD.assets/`

Use descriptive filenames such as:

- `cliff-motivation.png`
- `cliff-adjustment.png`
- `camera-perspective-explainer.png`

Expected outcome: every candidate image needed for editing already exists locally.

### Task 2: Select Final Images and Map Them to Formulas

**Files:**
- Read: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/DL/CV/Paper-Reading/docs/superpowers/specs/2026-04-17-cliff-formula-images-design.md`
- Read: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/DL/CV/Paper-Reading/assets/05-Kpts-OD.assets/`

- [ ] **Step 1: Score each candidate against the design criteria**

For each saved candidate, answer these questions explicitly:

- Does it explain `c_x`, `c_y`, `b`, or `f`?
- Does it explain `M_crop`, `M_full`, or the crop-to-full transform?
- Would removing it make the note meaningfully less understandable?

Expected outcome: a final set of 2-4 images only.

- [ ] **Step 2: Assign each final image to one formula block**

Prepare the final mapping:

- `I_{bbox}` block: 1-2 images
- crop-to-full translation block: 1-2 images

Do not assign one image to both formula blocks unless it truly explains both without causing duplication.

Expected outcome: placement decisions are fixed before editing the markdown.

- [ ] **Step 3: Draft the one-line Chinese explanation for each image**

Write one short explanation line per final image using this pattern:

- what this figure shows
- which formula it supports
- which symbol or relation it clarifies

Example pattern:

```text
这张图说明 bbox center 和 bbox size 都定义在 full frame 中，因此 `c_x,c_y,b` 不是 crop 内部量，而是把原图位置信息重新编码回回归器的输入。
```

Expected outcome: every image has its caption-like sentence ready before editing the note.

### Task 3: Update the Markdown Note

**Files:**
- Modify: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/DL/CV/Paper-Reading/05-Kpts-OD.md`

- [ ] **Step 1: Insert the selected image(s) under the `I_{bbox}` formula**

Edit the `Core Mechanism -> Architecture` subsection so that the formula block is followed by the selected local image embed(s) and their one-line explanation(s).

Use relative paths like:

```markdown
![CLIFF motivation](./assets/05-Kpts-OD.assets/cliff-motivation.png)

这张图说明 ...
```

Expected outcome: the `I_{bbox}` formula no longer stands alone and now has visual support.

- [ ] **Step 2: Insert the selected image(s) under the crop-to-full translation formula**

Edit the `Full-frame supervision` subsection so the conversion formula is followed by the selected local image embed(s) and one-line explanation(s).

Use relative paths like:

```markdown
![CLIFF adjustment](./assets/05-Kpts-OD.assets/cliff-adjustment.png)

这张图说明 ...
```

Expected outcome: the translation conversion formula now reads with visual geometry context.

- [ ] **Step 3: Keep the note compact and local-style consistent**

While editing, preserve all existing sections and text unless a tiny wording adjustment is needed to connect image and formula more naturally.

Hard limits:

- do not turn the section into a gallery
- do not add detached image dumps
- do not rewrite unrelated CLIFF paragraphs
- do not touch the Intro or PIPNet sections

Expected outcome: the diff stays tightly scoped to the existing CLIFF subsection.

### Task 4: Verify the Markdown Update

**Files:**
- Read: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/DL/CV/Paper-Reading/05-Kpts-OD.md`
- Read: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/DL/CV/Paper-Reading/assets/05-Kpts-OD.assets/`

- [ ] **Step 1: Verify every embedded path points to a real local file**

Check that every new markdown image reference under the CLIFF section corresponds to a file that exists under `assets/05-Kpts-OD.assets/`.

Expected outcome: zero broken local image links.

- [ ] **Step 2: Re-read the final CLIFF section for placement correctness**

Confirm all of the following:

- `I_{bbox}` images appear directly below the `I_{bbox}` formula
- crop-to-full images appear directly below the conversion formula
- each image has a short Chinese explanation
- the note still reads top-to-bottom without visual clutter

Expected outcome: visual placement matches the plan.

- [ ] **Step 3: Check scope and source integrity**

Confirm:

- only `05-Kpts-OD.md` and new asset files changed
- the chosen images are official or clearly attributable supplemental sources
- no weakly related benchmark/result-only images slipped in

Expected outcome: the update matches the approved design and stays trustworthy.

## Spec Coverage Check

- Official-first sourcing: covered in Task 1 Step 2
- Optional auxiliary blog/explainer sourcing: covered in Task 1 Step 3
- Local asset storage: covered in Task 1 Step 4 and Task 4 Step 1
- Formula-local placement: covered in Task 3 Steps 1-2
- One-line Chinese explanations: covered in Task 2 Step 3 and Task 3 Steps 1-2
- Compact scope and note integrity: covered in Task 3 Step 3 and Task 4 Step 3

## Placeholder Scan

No `TBD`, `TODO`, or implied follow-up placeholders remain. Each step names the exact target file, action, and expected outcome.

## Type / Naming Consistency Check

The plan consistently uses the same formula groups, file paths, asset directory, and note section names throughout. There is no mixed naming for the note, asset folder, or source-priority policy.
