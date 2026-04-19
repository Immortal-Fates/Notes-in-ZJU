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

- Failure handling: If any step (download, link resolution, or upload) fails, the note remains unchanged.

## Configuration

config.example.json is a template, not a runnable default. Copy it to a local machine-specific config path and rename to config.json. The system requires a valid config.json; if no valid config.json is found or the provided config is invalid, initialization fails with a clear error. Do not rely on placeholders.

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
 - Allowed: paper-id, note-name
 - Default: paper-id
 - Behavior: paper-id uses the internal paper identifier for the file name; note-name uses the note file stem (the base filename without extension) when available. If unresolved, falls back to paper-id. Note-name is not derived from the H1/frontmatter.
 - Validation: Invalid paper_name_policy values will cause initialization to fail with a descriptive error; there is no silent fallback.

 c) Supported remote_path_template placeholders and sanitization expectations
 - Placeholders: {note_name}, {paper_id}
 - Default: PaperReading/{note_name}/{paper_id}.pdf
 - Behavior: placeholders are expanded safely; path components are sanitized by replacing invalid filesystem characters with '_'.
 - Validation: Unknown placeholders in remote_path_template will cause initialization to fail with a descriptive error.

d) Exact config discovery path or precedence
- Discovery path: This skill's directory (the directory containing this README and config.json).
- Precedence: config.json is used if present; if not found, initialization will fail (no fallback to config.example.json).
- If config.json is missing altogether, initialization fails with a clear error message.

e) ~ expansion and fail-fast behavior for missing credential files
- Paths containing ~ are expanded to the user's home directory.
- If the expanded client_secret_file does not exist, initialization fails fast with a clear error.
- If the expanded token_file does not exist, first run will create it through the browser OAuth flow. Later runs will reuse and refresh it automatically when possible.
