# Google Drive Paper Upload

## What this skill does

When explicitly requested, this skill:
1. reads a target note
2. identifies the selected paper citation line
3. resolves a downloadable PDF URL
4. downloads the PDF to a temporary path outside the repo
5. uploads the PDF to Google Drive
6. inserts or updates `([My PDF](...))` on the citation line
7. deletes the temporary file

## Must do

- Run only on explicit user request.
- Never store the downloaded PDF inside `Notes-in-ZJU`.
- Patch only the citation line for the selected paper.
- Leave the note unchanged if any external step fails.

## Trigger, inputs and failure behavior (docs-only)

- Explicit trigger wording: Activate this skill using the exact trigger keyword
  "google-drive-paper-upload" followed by the target note path and, when needed,
  the paper identifier to disambiguate. Example (illustrative):
  google-drive-paper-upload /notes/2026/summary.md --paper-id "Paper Title"

- Script entrypoint: The runtime entrypoint is
  `python3 .agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py --note-path <path> --paper-id "<paper id>"`.
  Optionally pass `--config-path /absolute/or/repo-relative/config.json` to override
  the default skill-local `config.json`, and `--dry-run` to exercise the flow without
  writing the note back to disk.

- Required note path: The trigger must include the path to the note file that contains
  the target citation line to update. The path can be absolute or repo-relative.
- Paper identifier requirement: If the note contains multiple papers or the citation is
  ambiguous, include the paper identifier; otherwise the identifier is optional.
- Failure behavior: If any step (download, link resolution, or upload) fails, do not modify
  the note. The note remains exactly as it was prior to invocation.

## Must not do

- Do not auto-run during normal `paper-summary` use.
- Do not rewrite note body content.
- Do not commit credentials or tokens.
