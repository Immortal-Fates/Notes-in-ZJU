# Decisions

- Scope: repo-wide for eligible loose course/note sets.
- Ordering: preserve existing lecture/chapter order when evidenced; otherwise lexical fallback.
- Remote `https://...` images remain unchanged.
- Shared asset buckets are not split in pass 1.
- 2026-03-30: Task-4 rewriting uses lexical relative-target resolution plus explicit prefix-move maps so note moves and asset-root repoints can be simulated safely without rewriting real repo markdown during helper development.
- 2026-03-30: HTML `<img>` rewriting is constrained to the `src` attribute span only, preserving all non-path bytes (attribute order, quoting, `style`, `alt`, `width`, and surrounding markup) verbatim.
- 2026-03-30: Task 3 classifies any non-Windows URI scheme (for example `https://...` and `vscode-file://...`) as a non-failing URL, while pure `#anchor` targets remain a distinct non-failing anchor category.
- 2026-03-30: Task 3 reports alias buckets by watchlist path/basename and keeps them separate from non-alias shared buckets and single-owner buckets in baseline evidence.
- 2026-03-30: Task 6 legacy `markdown-img` execution keeps shared roots in place by default (no heuristic per-note bucket splitting, no implicit shared-root relocation) and only rewrites `src`/path targets through the task-4 safe rewriter when a source note actually moves.
- 2026-03-30: Task 7 alias/shared/cross-note execution accepts only explicit deterministic ownership mappings (`migration_class` D + non-`NN-` manifest leaf target + task-2 manual cohort ownership); non-deterministic alias cohorts (for example `math/assets/OR.assets`) remain unresolved debt.
- 2026-03-30: Task 7 keeps alias/shared bucket names in-place by default, records manifest-path drift as informational state, and preserves missing Windows absolute-path targets as debt rows instead of fabricating relative rewrites.
- 2026-03-30: Task-8 inbound-link rewrite implemented per manifest inbound-reference fields and post-migration audit outputs; updated root README.md and inbound-note READMEs as required; baseline debt for non-manifest-linked links preserved.
