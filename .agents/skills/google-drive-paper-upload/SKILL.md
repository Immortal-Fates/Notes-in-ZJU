# Google Drive Paper Upload

## What this skill does

When explicitly requested to upload one PDF, this skill:
1. reads a target note
2. identifies the selected paper citation line
3. resolves a downloadable PDF URL
4. downloads the PDF to a temporary path outside the repo
5. uploads the PDF to Google Drive using the paper's common name, such as
   `ATSS.pdf` or `transformer.pdf`
6. inserts or updates `([My PDF](...))` on the citation line
7. deletes the temporary file

When explicitly requested to rename existing Google Drive paper PDFs, this skill:
1. scans the target Markdown file or directory for citation-line `My PDF` links
2. infers each target common PDF name with the same naming rules
3. renames each existing Google Drive file
4. refreshes the corresponding `My PDF` link with the Drive API share link

## Must do

- Run only on explicit user request.
- Never store the downloaded PDF inside `Notes-in-ZJU`.
- Use short common PDF names, not full paper titles or note-derived paths. Prefer
  explicit aliases already written on the citation line after `--`, such as
  `-- ATSS`; otherwise pass `--file-name <common-name>` when the common name is
  not obvious from the citation line.
- Patch only the citation line for the selected paper.
- Leave the note unchanged if any external step fails.
- For existing-file renames, run `--rename-existing --dry-run` first and inspect
  the planned `file_id` -> `remote_name` operations before the real run.

## Trigger, inputs and failure behavior (docs-only)

- Explicit trigger wording: Activate this skill using the exact trigger keyword
  "google-drive-paper-upload" followed by the target note path and, when needed,
  the paper identifier to disambiguate. Example (illustrative):
  google-drive-paper-upload /notes/2026/summary.md --paper-id "Paper Title"

- Script entrypoint: The runtime entrypoint is
  `python3 .agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py --note-path <path> --paper-id "<paper id>"`.
  Add `--file-name "<common name>"` when the target Google Drive PDF name should
  be specified directly; the `.pdf` suffix is optional.
  Optionally pass `--config-path /absolute/or/repo-relative/config.json` to override
  auto-discovery in the skill directory and `~/.config/google-drive-paper-upload/config.json`,
  and `--dry-run` to exercise the flow without writing the note back to disk.

- Existing PDF rename entrypoint:
  `python3 .agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py --note-path . --note-root <file-or-directory> --rename-existing`.
  Add `--dry-run` first to print the planned operations without connecting to
  Google Drive or writing Markdown.

- Required note path: The trigger must include the path to the note file that contains
  the target citation line to update. The path can be absolute or repo-relative.
- Paper identifier requirement: If the note contains multiple papers or the citation is
  ambiguous, include the paper identifier; otherwise the identifier is optional.
- Failure behavior: If any step (download, link resolution, or upload) fails, do not modify
  the note. The note remains exactly as it was prior to invocation.

## Must not do

- Do not auto-run during normal `paper-summary` use.
- Do not upload files named with the full paper title when a common name is
  available or supplied.
- Do not rewrite note body content.
- Do not commit credentials or tokens.
