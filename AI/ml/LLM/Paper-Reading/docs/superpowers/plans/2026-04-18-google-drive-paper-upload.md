# Google Drive Paper Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repo-local skill under `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents` that, only when explicitly invoked, downloads a paper PDF to a temporary directory outside the repo, uploads it to Google Drive, and inserts or updates `([My PDF](...))` on the target paper citation line.

**Architecture:** The feature is split into a repo-local skill definition plus one focused Python helper script. The skill defines invocation rules and constraints, while the script handles note parsing, PDF URL resolution, temp-file download, Google Drive upload/link retrieval, markdown patching, and guaranteed cleanup. Configuration lives beside the skill as a non-secret example file so the workflow stays portable without storing credentials in notes.

**Tech Stack:** Repo-local OpenCode skill, Python 3, Google Drive API, markdown text patching, temporary filesystem storage.

---

### Task 1: Create the skill directory and top-level skill definition

**Files:**
- Create: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/SKILL.md`
- Create: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/README.md`

- [ ] **Step 1: Write the skill definition skeleton**

```md
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

## Must not do

- Do not auto-run during normal `paper-summary` use.
- Do not rewrite note body content.
- Do not commit credentials or tokens.
```

- [ ] **Step 2: Run a quick read check on the new skill file**

Run: `python - <<'PY'
from pathlib import Path
path = Path('/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/SKILL.md')
print(path.exists(), path.read_text()[:80])
PY`
Expected: `True` followed by the beginning of the skill file

- [ ] **Step 3: Write the skill README with setup intent**

```md
# Google Drive Paper Upload Skill

This repo-local skill uploads a paper PDF to Google Drive only when explicitly requested.

It is designed for notes in `Notes-in-ZJU` that already contain paper citation lines such as:

- `- **Paper Title**... ([Paper](...))`

The implementation downloads to a temporary location outside the repo, uploads to Google Drive, then inserts or updates:

- `([My PDF](...))`

on the same citation line.
```

- [ ] **Step 4: Run a quick read check on the README**

Run: `python - <<'PY'
from pathlib import Path
path = Path('/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/README.md')
print(path.exists(), path.read_text().splitlines()[0])
PY`
Expected: `True # Google Drive Paper Upload Skill`

- [ ] **Step 5: Commit**

```bash
git add /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/SKILL.md /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/README.md
git commit -m "feat: add Google Drive paper upload skill scaffold"
```

### Task 2: Add configuration template and configuration loader contract

**Files:**
- Create: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/config.example.json`
- Modify: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/README.md`

- [ ] **Step 1: Write the configuration template**

```json
{
  "drive_folder_id": "YOUR_GOOGLE_DRIVE_FOLDER_ID",
  "link_type": "webViewLink",
  "paper_name_policy": "paper-id",
  "remote_path_template": "PaperReading/{note_name}/{paper_id}.pdf",
  "client_secret_file": "~/.config/google-drive-paper-upload/client_secret.json",
  "token_file": "~/.config/google-drive-paper-upload/token.json"
}
```

- [ ] **Step 2: Document configuration usage in the README**

```md
## Configuration

Copy `config.example.json` to a local machine-specific config path before first use.

Required values:
- `drive_folder_id`
- `client_secret_file`
- `token_file`

Do not commit populated credential files into `Notes-in-ZJU`.
```

- [ ] **Step 3: Validate the example JSON parses cleanly**

Run: `python - <<'PY'
import json
from pathlib import Path
path = Path('/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/config.example.json')
print(json.loads(path.read_text())['drive_folder_id'])
PY`
Expected: `YOUR_GOOGLE_DRIVE_FOLDER_ID`

- [ ] **Step 4: Commit**

```bash
git add /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/config.example.json /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/README.md
git commit -m "docs: add Google Drive upload skill configuration template"
```

### Task 3: Implement note parsing and citation-line selection

**Files:**
- Create: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py`
- Test with: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/LLM/Paper-Reading/02-1-GPT.md`

- [ ] **Step 1: Write the first failing test logic as an executable script snippet**

```python
from pathlib import Path
from upload_paper_to_gdrive import find_paper_entry_line

note_path = Path('/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/LLM/Paper-Reading/02-1-GPT.md')
text = note_path.read_text()
line = find_paper_entry_line(text, 'Improving Language Understanding by Generative Pre-Training')
assert '[Paper](' in line
assert 'Improving Language Understanding by Generative Pre-Training' in line
```

- [ ] **Step 2: Run the test snippet and verify it fails**

Run: `python /tmp/test_find_paper_entry_line.py`
Expected: FAIL with `ImportError` or `AttributeError` because the function does not exist yet

- [ ] **Step 3: Write the minimal parsing implementation**

```python
def find_paper_entry_line(note_text: str, paper_identifier: str) -> str:
    for line in note_text.splitlines():
        if paper_identifier in line and line.lstrip().startswith('- '):
            return line
    raise ValueError(f'Could not find paper entry for: {paper_identifier}')


def replace_or_append_my_pdf(line: str, new_url: str) -> str:
    marker = '([My PDF]('
    if marker in line:
        prefix, rest = line.split(marker, 1)
        _, suffix = rest.split('))', 1)
        return f"{prefix}([My PDF]({new_url})){suffix}"
    return f"{line} ([My PDF]({new_url}))"
```

- [ ] **Step 4: Re-run the parsing test snippet**

Run: `python /tmp/test_find_paper_entry_line.py`
Expected: PASS with no output

- [ ] **Step 5: Add a second executable check for in-place update behavior**

```python
from upload_paper_to_gdrive import replace_or_append_my_pdf

line = '- **Paper**. Author. **arXiv**, **2024**, ([Paper](https://example.com)) ([My PDF](https://old.example))'
updated = replace_or_append_my_pdf(line, 'https://new.example')
assert updated.count('([My PDF](') == 1
assert 'https://new.example' in updated
assert 'https://old.example' not in updated
```

- [ ] **Step 6: Commit**

```bash
git add /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py
git commit -m "feat: add citation line parsing for paper upload skill"
```

### Task 4: Implement PDF URL resolution and temporary download

**Files:**
- Modify: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py`
- Test with: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/LLM/Paper-Reading/02-1-GPT.md`

- [ ] **Step 1: Write the failing URL-resolution test snippet**

```python
from upload_paper_to_gdrive import extract_download_url

line = '- **BERT**. ... ([Arxiv](https://arxiv.org/abs/1810.04805)) ([Code](https://github.com/google-research/bert))'
assert extract_download_url(line) == 'https://arxiv.org/pdf/1810.04805.pdf'
```

- [ ] **Step 2: Run the URL-resolution test and verify it fails**

Run: `python /tmp/test_extract_download_url.py`
Expected: FAIL with `ImportError` or `AttributeError`

- [ ] **Step 3: Implement URL resolution and temp download**

```python
import re
import tempfile
import urllib.request
from pathlib import Path


def extract_download_url(line: str) -> str:
    pdf_match = re.search(r'\((https?://[^)]+\.pdf)\)', line)
    if pdf_match:
        return pdf_match.group(1)

    arxiv_match = re.search(r'https?://arxiv\.org/abs/([0-9]+\.[0-9]+)(v\d+)?', line)
    if arxiv_match:
        paper_id = arxiv_match.group(1)
        return f'https://arxiv.org/pdf/{paper_id}.pdf'

    generic_match = re.search(r'\((https?://[^)]+)\)', line)
    if generic_match:
        return generic_match.group(1)

    raise ValueError('Could not resolve a downloadable paper URL from citation line')


def download_pdf_to_temp(download_url: str, filename_hint: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix='paper-upload-'))
    pdf_path = temp_dir / f'{filename_hint}.pdf'
    urllib.request.urlretrieve(download_url, pdf_path)
    return pdf_path
```

- [ ] **Step 4: Re-run the URL-resolution test**

Run: `python /tmp/test_extract_download_url.py`
Expected: PASS with no output

- [ ] **Step 5: Add an executable smoke test for temp download cleanup contract**

```python
from pathlib import Path
from upload_paper_to_gdrive import download_pdf_to_temp

path = download_pdf_to_temp('https://arxiv.org/pdf/1810.04805.pdf', 'bert-test')
assert path.exists()
assert '/Notes-in-ZJU/' not in str(path)
assert path.suffix == '.pdf'
print(path)
```

- [ ] **Step 6: Commit**

```bash
git add /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py
git commit -m "feat: add PDF download resolution for paper upload skill"
```

### Task 5: Implement Google Drive upload and link retrieval

**Files:**
- Modify: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py`
- Modify: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/README.md`

- [ ] **Step 1: Write the failing upload contract test snippet with a stubbed client**

```python
from pathlib import Path
from upload_paper_to_gdrive import upload_pdf_and_get_link


class FakeDriveClient:
    def upload_pdf(self, pdf_path: Path, remote_name: str) -> str:
        assert pdf_path.name == 'paper.pdf'
        assert remote_name == '1810.04805.pdf'
        return 'file-123'

    def get_share_link(self, file_id: str) -> str:
        assert file_id == 'file-123'
        return 'https://drive.google.com/file/d/file-123/view'


link = upload_pdf_and_get_link(FakeDriveClient(), Path('paper.pdf'), '1810.04805.pdf')
assert link == 'https://drive.google.com/file/d/file-123/view'
```

- [ ] **Step 2: Run the upload contract test and verify it fails**

Run: `python /tmp/test_upload_pdf_and_get_link.py`
Expected: FAIL with `ImportError` or `AttributeError`

- [ ] **Step 3: Implement the minimal upload orchestration boundary**

```python
def upload_pdf_and_get_link(drive_client, pdf_path: Path, remote_name: str) -> str:
    file_id = drive_client.upload_pdf(pdf_path, remote_name)
    return drive_client.get_share_link(file_id)
```

- [ ] **Step 4: Add the real Google Drive client wrapper**

```python
class GoogleDriveClient:
    def __init__(self, service, drive_folder_id: str):
        self.service = service
        self.drive_folder_id = drive_folder_id

    def upload_pdf(self, pdf_path: Path, remote_name: str) -> str:
        ...

    def get_share_link(self, file_id: str) -> str:
        ...
```

Implementation details for this step:
- use the configured folder id from config
- create or update a single PDF file for the same logical paper name
- retrieve one stable Google Drive link shape consistently
- raise explicit exceptions on upload or permission failures

- [ ] **Step 5: Re-run the stubbed upload contract test**

Run: `python /tmp/test_upload_pdf_and_get_link.py`
Expected: PASS with no output

- [ ] **Step 6: Document required Google libraries in the README**

```md
## Python dependencies

Install the Google Drive client dependencies before first use:

```bash
pip install google-api-python-client google-auth google-auth-oauthlib
```
```

- [ ] **Step 7: Commit**

```bash
git add /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/README.md
git commit -m "feat: add Google Drive upload client for paper upload skill"
```

### Task 6: Implement note patching, CLI entrypoint, and cleanup guarantees

**Files:**
- Modify: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py`
- Modify: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/SKILL.md`

- [ ] **Step 1: Write the failing end-to-end dry-run test snippet with a fake drive client**

```python
from pathlib import Path
from upload_paper_to_gdrive import process_note_entry


class FakeDriveClient:
    def upload_pdf(self, pdf_path, remote_name):
        return 'file-123'

    def get_share_link(self, file_id):
        return 'https://drive.google.com/file/d/file-123/view'


note_path = Path('/tmp/paper-note.md')
note_path.write_text('- **BERT**. Test. ([Arxiv](https://arxiv.org/abs/1810.04805))\n')
process_note_entry(note_path, 'BERT', FakeDriveClient(), dry_run=False)
text = note_path.read_text()
assert '([My PDF](https://drive.google.com/file/d/file-123/view))' in text
assert text.count('([My PDF](') == 1
```

- [ ] **Step 2: Run the end-to-end dry-run test and verify it fails**

Run: `python /tmp/test_process_note_entry.py`
Expected: FAIL because `process_note_entry` does not exist yet

- [ ] **Step 3: Implement the orchestration function with `try/finally` cleanup**

```python
def process_note_entry(note_path: Path, paper_identifier: str, drive_client, dry_run: bool = False) -> str:
    note_text = note_path.read_text()
    line = find_paper_entry_line(note_text, paper_identifier)
    download_url = extract_download_url(line)
    temp_pdf_path = None
    try:
        temp_pdf_path = download_pdf_to_temp(download_url, paper_identifier.replace(' ', '-').lower())
        link = upload_pdf_and_get_link(drive_client, temp_pdf_path, temp_pdf_path.name)
        new_line = replace_or_append_my_pdf(line, link)
        new_text = note_text.replace(line, new_line, 1)
        if not dry_run:
            note_path.write_text(new_text)
        return link
    finally:
        if temp_pdf_path is not None and temp_pdf_path.exists():
            temp_pdf_path.unlink()
            temp_pdf_path.parent.rmdir()
```

- [ ] **Step 4: Add the CLI entrypoint**

```python
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--note-path', required=True)
    parser.add_argument('--paper-id', required=True)
    parser.add_argument('--config-path', required=True)
    args = parser.parse_args()
    ...
```

CLI requirements for this step:
- load config before API initialization
- initialize the real Google Drive client
- print the final link on success
- exit non-zero on failure

- [ ] **Step 5: Re-run the end-to-end dry-run test**

Run: `python /tmp/test_process_note_entry.py`
Expected: PASS with no output

- [ ] **Step 6: Add explicit skill invocation instructions to `SKILL.md`**

```md
## Invocation pattern

Use this skill only when the user explicitly asks to upload a paper PDF to Google Drive and add or refresh `My PDF` in a note.

Required inputs:
- note path
- paper identifier when the note has multiple papers
```

- [ ] **Step 7: Commit**

```bash
git add /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/SKILL.md
git commit -m "feat: add end-to-end paper upload orchestration"
```

### Task 7: Verify against the real note format and protect against regressions

**Files:**
- Test: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/LLM/Paper-Reading/02-1-GPT.md`
- Test: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/LLM/Paper-Reading/02-3-BERT.md`
- Modify if needed: `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py`

- [ ] **Step 1: Run a dry-run parse check on a direct PDF citation**

Run: `python /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py --note-path /home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/LLM/Paper-Reading/02-1-GPT.md --paper-id "Improving Language Understanding by Generative Pre-Training" --config-path /path/to/local-config.json --dry-run`
Expected: prints a resolved link or a dry-run success message without modifying the note

- [ ] **Step 2: Run a dry-run parse check on an arXiv citation**

Run: `python /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py --note-path /home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/LLM/Paper-Reading/02-3-BERT.md --paper-id "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" --config-path /path/to/local-config.json --dry-run`
Expected: resolves `https://arxiv.org/pdf/1810.04805.pdf` internally and reports dry-run success

- [ ] **Step 3: Run a non-destructive citation patch check on a temp copy**

Run: `cp /home/immortal-pc1/ws/docs/Notes-in-ZJU/AI/ml/LLM/Paper-Reading/02-3-BERT.md /tmp/02-3-BERT.test.md && python /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py --note-path /tmp/02-3-BERT.test.md --paper-id "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" --config-path /path/to/local-config.json`
Expected: exactly one `([My PDF](...))` appears on the citation line and the rest of the file remains unchanged

- [ ] **Step 4: Run a duplicate-protection check on the temp copy**

Run: `python /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py --note-path /tmp/02-3-BERT.test.md --paper-id "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" --config-path /path/to/local-config.json`
Expected: still only one `([My PDF](...))` link exists after the second run

- [ ] **Step 5: Run a cleanup check**

Run: `python - <<'PY'
from pathlib import Path
for path in Path('/tmp').glob('paper-upload-*'):
    print(path)
PY`
Expected: no leftover temp directories from successful runs

- [ ] **Step 6: Commit**

```bash
git add /home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents/skills/google-drive-paper-upload/scripts/upload_paper_to_gdrive.py
git commit -m "test: verify Google Drive paper upload skill against note formats"
```

## Self-Review

- Spec coverage: the plan covers explicit opt-in invocation, transient PDF download, Google Drive upload, note patching, cleanup, repo-local placement under `/home/immortal-pc1/ws/docs/Notes-in-ZJU/.agents`, and idempotent `My PDF` updates.
- Placeholder scan: there are no `TBD` or `TODO` placeholders in the plan; every task has exact files and concrete commands.
- Type consistency: the plan consistently uses `find_paper_entry_line`, `replace_or_append_my_pdf`, `extract_download_url`, `download_pdf_to_temp`, `upload_pdf_and_get_link`, and `process_note_entry` across tasks.
