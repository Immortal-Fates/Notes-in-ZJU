# Google Drive Paper Upload Skill

This repo-local skill uploads a paper PDF to Google Drive only when explicitly requested.

It is designed for notes in `Notes-in-ZJU` that already contain paper citation lines such as:

- `- **Paper Title**... ([Paper](...))`

- Trigger and inputs:
  The skill activates on an explicit trigger using the exact keyword "google-drive-paper-upload" followed by the target note path. If the note contains multiple papers or the citation is ambiguous, include the paper identifier to disambiguate.
- Required note path: Provide the path to the note file that contains the citation line to update (absolute or repo-relative).
- Paper identifier usage: Include the paper identifier only if needed to disambiguate among multiple papers in the note.

- Behavior: The implementation downloads to a temporary location outside the repo, uploads to Google Drive, then inserts or updates:

- `([My PDF](...))`

- on the same citation line.

- Upload name: By default, the Google Drive PDF is named with the paper's short
  common name, not the full paper title and not a note-derived path. If the
  citation line ends with an explicit alias such as `-- ATSS`, the uploaded file
  is named `ATSS.pdf`. If the common name is not written in the note, pass
  `--file-name transformer`; the `.pdf` suffix is optional.

- Existing Drive PDFs can be renamed with:
  `python3 .agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py --note-path . --note-root <file-or-directory> --rename-existing`.
  Run the same command with `--dry-run` first to print the planned `file_id` to
  common-name PDF mapping without contacting Google Drive or changing Markdown.

- Failure handling: If any step (download, link resolution, or upload) fails, the note remains unchanged.

## Configuration

config.example.json is a template, not a runnable default. Copy it to a local machine-specific config path and rename to config.json. The system requires a valid config.json; if no valid config.json is found or the provided config is invalid, initialization fails with a clear error. Do not rely on placeholders.

Default discovery order:
- skill-local `config.json`
- `~/.config/google-drive-paper-upload/config.json`

If both exist, the skill-local config wins. Passing `--config-path` overrides both.

Required values:
- `drive_folder_id`
- `client_secret_file`
- `token_file`

On first run, `token_file` may point to a file that does not exist yet. The script will open a browser-based Google OAuth flow, then create that file automatically after successful authorization.

Do not commit populated credential files into `Notes-in-ZJU`.

## Python dependencies

Install the Google Drive client dependencies before first use:

```bash
pip install google-api-python-client google-auth google-auth-oauthlib
```

## Configuration documentation (Task 2 quality review)

a) Allowed link_type values and default
- Allowed: webViewLink, webContentLink
- Default: webViewLink
- Behavior: webViewLink yields a shareable viewer link; webContentLink yields a direct download/link.
- Validation: Invalid link_type values will cause initialization to fail with a descriptive error; there is no silent fallback.

 b) Allowed paper_name_policy values and exact behavior
 - Allowed: common-name, paper-id, note-name
 - Default: common-name
 - Behavior: common-name uses `--file-name` when supplied, otherwise an explicit
   citation-line alias such as `-- ATSS`, otherwise a configured title override
   such as `Attention Is All You Need` -> `transformer`, and finally the paper
   identifier as a fallback. paper-id uses the internal paper identifier for the
   file name; note-name uses the note file stem (the base filename without
   extension) when available. Note-name is not derived from the H1/frontmatter.
 - Validation: Invalid paper_name_policy values will cause initialization to fail with a descriptive error; there is no silent fallback.

 c) Supported remote_path_template placeholders and sanitization expectations
 - Placeholders: {note_name}, {paper_id}, {paper_name}
 - Default: {paper_name}.pdf
 - Behavior: placeholders are expanded safely; path components are sanitized by replacing invalid filesystem characters with '_'.
 - Validation: Unknown placeholders in remote_path_template will cause initialization to fail with a descriptive error.

 c2) common_name_overrides
 - Optional JSON object mapping exact paper titles to short common names.
 - Default includes `Attention Is All You Need` -> `transformer`,
   `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
   -> `ViT`, `Swin Transformer: Hierarchical Vision Transformer using Shifted
   Windows` -> `Swin-Transformer`, and `Masked Autoencoders Are Scalable Vision
   Learners` -> `MAE`.
 - Values may include or omit `.pdf`; the upload name is still rendered through
   `remote_path_template`.

 c3) Legacy default config migration
 - A config that still contains the old default pair `paper_name_policy:
   paper-id` and `remote_path_template: PaperReading/{note_name}/{paper_id}.pdf`
   is treated as the current default pair `common-name` and `{paper_name}.pdf`.
 - This prevents old copied template configs from continuing to upload files with
   full paper-title paths after the naming policy changed.

d) Exact config discovery path or precedence
- Discovery paths, in order:
  1. This skill's directory (the directory containing this README and `config.json`)
  2. `~/.config/google-drive-paper-upload/config.json`
- Precedence: skill-local `config.json` wins when both are present. Passing `--config-path` overrides both defaults.
- If no discovered config exists, initialization fails with a clear error message.

e) ~ expansion and fail-fast behavior for missing credential files
- Paths containing ~ are expanded to the user's home directory.
- If the expanded client_secret_file does not exist, initialization fails fast with a clear error.
- If the expanded token_file does not exist, first run will create it through the browser OAuth flow. Later runs will reuse and refresh it automatically when possible.
