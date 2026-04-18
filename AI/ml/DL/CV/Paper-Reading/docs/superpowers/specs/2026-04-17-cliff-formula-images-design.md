# CLIFF Formula Images Design

> For this note update, the goal is to add externally sourced explanatory images to the CLIFF section so the formulas read like a guided derivation instead of isolated math blocks.

## Goal

Enhance `05-Kpts-OD.md` by adding 2-4 web-sourced images that visually explain where the two core CLIFF formulas come from:

1. `I_{bbox} = [c_x / f_{CLIFF}, c_y / f_{CLIFF}, b / f_{CLIFF}]`
2. the crop-to-full translation conversion

The result should stay compact, preserve the existing markdown structure, and make each formula easier to understand during later review.

## Scope

In scope:

- Search for high-quality existing images online
- Prefer official paper figures and official repository figures
- Allow one or two blog/explainer images only as auxiliary material when they directly clarify the geometry better than the paper figure alone
- Save selected images locally under `assets/05-Kpts-OD.assets/`
- Insert the images directly below the formula they explain
- Add one short Chinese explanation line under each image explaining which variable or geometric relation it clarifies

Out of scope:

- Drawing new diagrams from scratch
- Restructuring the whole CLIFF note
- Rewriting unrelated sections of `05-Kpts-OD.md`
- Using low-quality or weakly related “result showcase” images that do not explain the formulas

## Source Policy

Image source priority:

1. CLIFF paper figures
2. CLIFF official repository figures
3. High-quality third-party explanatory figures from blogs or technical posts

Third-party images are acceptable only when all of the following hold:

- they explain the exact geometry behind the formula more clearly than the official figures alone
- they are from a named, technically credible source
- they do not contradict the paper’s notation or geometry
- they are used as supplemental intuition, not as the primary authority

If a third-party image seems visually nice but does not map clearly to `c_x`, `c_y`, `b`, `f`, `M_crop`, `M_full`, or the projection path, it should be rejected.

## Image Plan

### Group A: `I_{bbox}` formula

Purpose: explain why the bbox feature vector encodes full-frame location and size.

Target image types:

- a CLIFF motivation or geometry figure showing the full image, crop, and viewpoint relationship
- an auxiliary perspective-geometry or camera-coordinate explanation image, if needed, to make `c_x / f` and `c_y / f` intuitive

What the final note should clarify near this formula:

- `c_x`, `c_y` are bbox-center offsets relative to the full-image center
- `b` is the bbox size in the original image
- dividing by `f_{CLIFF}` makes the quantities geometrically meaningful and normalized to camera scale
- the feature vector carries discarded location information from the full frame back into the regressor

Expected count: 1-2 images

### Group B: crop-to-full translation conversion

Purpose: explain why root translation must be converted from the crop camera to the full-image camera before full-frame reprojection loss is computed.

Target image types:

- the CLIFF adjustment / camera-relation figure from the paper if readable
- one auxiliary explainer image if necessary to clarify crop camera vs full camera, bbox scale, and projection geometry

What the final note should clarify near this formula:

- `M_crop` and `M_full` are different camera coordinate references
- translation terms depend on bbox offset and bbox size because the crop changes the reference frame
- the converted translation is needed so `J_{2D}^{full}` is projected in the original full image instead of only the crop

Expected count: 1-2 images

## Placement in the Note

Only touch the CLIFF subsection in `05-Kpts-OD.md`.

Planned placement:

- under the `I_{bbox}` formula block: insert the selected image(s), then one short explanation line
- under the crop-to-full translation formula block: insert the selected image(s), then one short explanation line

Do not create a detached gallery section. The images must stay next to the formula they explain.

## Asset Rules

- Save all new files under `assets/05-Kpts-OD.assets/`
- Use descriptive filenames when possible, such as `cliff-motivation.png`, `cliff-adjustment.png`, `camera-perspective-explainer.png`
- Use local relative markdown paths like `./assets/05-Kpts-OD.assets/<file>`
- Prefer stable local copies over hotlinking

## Selection Criteria

Each selected image must pass at least one of these tests:

- directly explains one symbol or coordinate relation in a formula
- clarifies the crop-to-full camera transformation
- makes the reprojection path easier to understand than text alone

Each selected image must also avoid these failure modes:

- only qualitative result visualization with no formula relevance
- unreadable labels after local embedding
- mismatched notation that would confuse the note
- duplicate explanatory value with another selected image

## Verification Plan

Before implementation is considered complete, verify:

- every new image is stored locally under `assets/05-Kpts-OD.assets/`
- every new image is embedded under the intended formula block
- every embedded image has a one-line explanation tied to the formula
- no unrelated sections in `05-Kpts-OD.md` are modified
- all source choices are traceable back to either the paper, the official repo, or a clearly identified explainer source

## Notes From Self-Review

- No placeholder sections remain
- Scope is limited to the existing CLIFF note, not the broader repository
- The design explicitly prefers official material and only uses blog images as supplemental intuition
- The outcome is concrete enough to convert into an implementation plan without further structural decisions
