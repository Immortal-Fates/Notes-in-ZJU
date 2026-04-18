# Google Drive Paper Upload Design

> For this workflow, the goal is to add an explicit opt-in skill that downloads a paper PDF into a temporary location outside `Notes-in-ZJU`, uploads it to Google Drive, and patches the note with a `([My PDF](...))` link on the citation line only.

## Goal

Provide a repo-local skill that performs a one-shot paper publishing flow when explicitly requested:

1. read a target paper note
2. find the selected paper entry and its source paper URL
3. download the PDF to a temporary directory outside the repository
4. upload the PDF to Google Drive
5. obtain a shareable Google Drive link
6. insert or update `([My PDF](...))` on the paper's metadata line
7. delete the temporary PDF

The repository should keep only markdown notes and explicitly retained local assets. The uploaded PDF must never be stored inside the repo as part of this workflow.

## Scope

In scope:

- an explicit skill for upload-on-demand only
- reading a paper URL from an existing markdown note entry
- downloading the PDF into a system temp directory
- uploading the PDF into a configured Google Drive folder
- creating or reusing a shareable link suitable for note insertion
- patching only the citation line of the selected paper entry with `([My PDF](...))`
- cleaning up temporary files after success or failure
- leaving the note unchanged when any external step fails

Out of scope:

- automatic upload during normal `paper-summary` execution
- storing the PDF inside `Notes-in-ZJU`
- rewriting the paper body sections such as `Takeaway`, `Motivation`, or `Core Mechanism`
- bulk-uploading every paper in a note
- syncing annotations back from Google Drive into the local machine

## User Interaction Model

This workflow is opt-in only.

The skill runs only when the user explicitly asks for upload behavior. It must not upload implicitly when summarizing or updating notes.

Recommended invocation intent:

- upload the current paper PDF to Google Drive and add `My PDF`
- upload paper for this note entry
- refresh the Google Drive link for this paper

The skill should accept either:

- a note path plus a paper identifier inside the note, or
- a note path when the target note contains only one relevant paper entry

If multiple paper entries exist and the target is ambiguous, the skill should stop and ask the user which paper entry to use instead of guessing.

## Source-of-Truth Model

- The markdown note is the source of truth for the paper entry and final inserted link.
- The source PDF is fetched transiently from the paper URL found in the note.
- The temporary file exists only long enough to complete upload.
- Google Drive is the remote storage backend for user access, not the canonical metadata source.

This means repeated runs should be able to reconstruct the workflow purely from the note plus configuration, without relying on committed local PDFs.

## Note Discovery Rules

The skill should follow the established paper-entry pattern already present in files such as:

- `AI/ml/LLM/Paper-Reading/02-1-GPT.md:7`
- `AI/ml/LLM/Paper-Reading/02-1-GPT.md:75`
- `AI/ml/LLM/Paper-Reading/02-3-BERT.md:7`

The target patch location is the first citation line of the paper entry, which already contains links like `([Paper](...))`, `([Arxiv](...))`, or `([Code](...))`.

Patch behavior:

- if `([My PDF](...))` already exists on that line, replace only its URL
- if it does not exist, append `([My PDF](...))` after the existing paper links on the same line
- do not edit any later body content in the paper entry
- do not reformat unrelated link text or punctuation unless needed to add the new link cleanly

## Paper URL Resolution

The skill must extract a download source from the existing citation line.

Preferred resolution order:

1. direct PDF URL already present in the note
2. arXiv abstract URL converted to the canonical PDF URL
3. other clearly downloadable paper URL already present in the note

If the note contains only a landing page with no deterministic PDF derivation and no direct downloadable URL can be found, the skill should stop and report that it could not identify a downloadable PDF source.

## Temporary Download Rules

- Download into a system temporary directory outside the repository.
- Use a deterministic temporary filename derived from the paper id or slug when possible.
- Never place the transient PDF under `Notes-in-ZJU`.
- Always delete the temporary file before exit, regardless of success or failure.

Temporary directories may be created per run, but they should be cleaned up immediately after use.

## Google Drive Upload Model

Google Drive should be accessed through the Google Drive API, not through a synced desktop folder.

Required remote behavior:

- upload the downloaded PDF to a configured destination folder in Google Drive
- use a deterministic remote naming rule such as `<paper-id>.pdf` or `<paper-slug>.pdf`
- prefer updating or replacing the same logical file path on repeated runs instead of creating endless duplicates
- return a shareable link suitable for insertion into markdown notes

Recommended remote layout:

- `PaperReading/<note-name>/<paper-id>.pdf`

This keeps the Drive structure understandable while still grouping files by note.

## Sharing Model

The generated link should be intended for the user's Google account workflow and inserted into the note as `My PDF`.

Required behavior:

- create or retrieve a stable Google Drive link after upload
- insert the returned link into the note as `([My PDF](...))`

The implementation may choose `webViewLink` or another stable Drive URL shape, but the chosen format should remain consistent across runs.

If the chosen Google Drive permission model cannot produce a usable link for the current account or folder policy, the skill should fail without modifying the note.

## Failure and Rollback Rules

This workflow has external side effects, so it must fail conservatively.

Required safety rules:

- if note parsing fails, do nothing
- if paper URL resolution fails, do nothing
- if download fails, do nothing
- if Google Drive upload fails, do nothing
- if link retrieval fails, do nothing
- if note patching fails after upload succeeds, report the upload result but do not partially rewrite unrelated note content
- always clean up temporary files

The note must only be modified after all external steps have completed successfully and the final link is available.

## Idempotency Rules

Repeated runs on the same paper should converge cleanly.

Expected behavior:

- the same paper entry remains a single citation line with at most one `My PDF` link
- rerunning the skill updates the existing `My PDF` URL instead of adding duplicates
- remote naming should make it practical to refresh the stored PDF for the same paper

## Recommended Skill Boundary

This should be a separate skill instead of default `paper-summary` behavior.

Reasoning:

- `paper-summary` is a note-writing workflow
- paper upload is an external publishing/storage action
- combining them by default would make normal note updates unexpectedly stateful and side-effectful

If later desired, `paper-summary` may call this upload skill only as an explicitly requested optional follow-up step.

## Configuration Requirements

The implementation should expect a small local configuration layer outside committed note content.

Configuration should cover:

- Google Drive authentication method
- destination Drive folder id or root folder rule
- preferred link format
- optional paper naming policy

Sensitive credentials must not be written into the markdown notes or committed into the repository.

## Verification Plan

Before implementation is considered complete, verify all of the following:

- the workflow does not create PDFs under `Notes-in-ZJU`
- the temporary PDF is deleted after the run
- the target note gains exactly one `([My PDF](...))` link on the intended citation line
- rerunning the workflow updates the existing `My PDF` link instead of duplicating it
- body sections such as `Takeaway` and `Core Mechanism` remain unchanged
- failure in download or upload leaves the note unchanged
- ambiguous multi-paper notes are rejected unless the target entry is specified clearly

## Notes From Self-Review

- The design now matches the user's corrected intent: transient download only, no repo-local PDF storage.
- Upload is explicitly opt-in and decoupled from normal paper-summary runs.
- The patch surface is narrow and tied to the repo's existing citation-line convention.
- External side effects are isolated so note edits happen only after a successful Google Drive link is ready.
