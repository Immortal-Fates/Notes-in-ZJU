import argparse
import json
from pathlib import Path
import re
import shutil
import string
import sys
import tempfile
from urllib.parse import urlparse
import urllib.request


class GoogleDriveUploadError(RuntimeError):
    pass


class GoogleDrivePermissionError(RuntimeError):
    pass


SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = SKILL_DIR / "config.json"
USER_CONFIG_DIR = Path.home() / ".config" / "google-drive-paper-upload"
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.json"
DEFAULT_LINK_TYPE = "webViewLink"
LEGACY_DEFAULT_PAPER_NAME_POLICY = "paper-id"
LEGACY_DEFAULT_REMOTE_PATH_TEMPLATE = "PaperReading/{note_name}/{paper_id}.pdf"
DEFAULT_PAPER_NAME_POLICY = "common-name"
DEFAULT_REMOTE_PATH_TEMPLATE = "{paper_name}.pdf"
DEFAULT_COMMON_NAME_OVERRIDES = {
    "attention is all you need": "transformer",
    "an image is worth 16x16 words: transformers for image recognition at scale": "ViT",
    "swin transformer: hierarchical vision transformer using shifted windows": "Swin-Transformer",
    "masked autoencoders are scalable vision learners": "MAE",
}
ALLOWED_LINK_TYPES = {"webViewLink", "webContentLink"}
ALLOWED_PAPER_NAME_POLICIES = {"common-name", "paper-id", "note-name"}


def _escape_drive_query_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _get_http_status(error: Exception):
    response = getattr(error, "resp", None)
    return getattr(response, "status", None)


def _pick_existing_drive_file(existing_files):
    if not existing_files:
        return None

    return sorted(
        existing_files,
        key=lambda item: (
            item.get("createdTime") or "",
            item.get("modifiedTime") or "",
            item.get("id") or "",
        ),
    )[0]


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip()
    return sanitized or "_"


def _validate_template_placeholders(template: str) -> None:
    allowed_fields = {"note_name", "paper_id", "paper_name"}
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name is None:
            continue
        if field_name not in allowed_fields:
            raise ValueError(
                "remote_path_template contains unsupported placeholder "
                f"{field_name!r}; allowed placeholders are {sorted(allowed_fields)!r}"
            )


def _strip_pdf_extension(value: str) -> str:
    stripped = value.strip()
    if stripped.lower().endswith(".pdf"):
        return stripped[:-4]
    return stripped


def _extract_paper_title_from_entry_line(line: str) -> str | None:
    match = re.match(r"\s*-\s+(?:\*\*|__)(.+?)(?:\*\*|__)", line)
    if not match:
        return None
    return match.group(1).strip().strip(" .") or None


def _extract_common_name_alias_from_entry_line(line: str) -> str | None:
    line_without_my_pdf = re.sub(r"\s*\(?\[My PDF\]\([^)]+\)\)?", "", line).rstrip()
    match = re.search(r"--\s*([^()\[\]\n]+?)\s*$", line_without_my_pdf)
    if not match:
        return None
    return match.group(1).strip().strip(" .") or None


def _normalize_common_name_overrides(raw_overrides) -> dict[str, str]:
    if raw_overrides is None:
        raw_overrides = DEFAULT_COMMON_NAME_OVERRIDES
    if not isinstance(raw_overrides, dict):
        raise ValueError("common_name_overrides must be a JSON object")

    normalized = {}
    for raw_title, raw_common_name in raw_overrides.items():
        if not isinstance(raw_title, str) or not raw_title.strip():
            raise ValueError("common_name_overrides keys must be non-empty strings")
        if not isinstance(raw_common_name, str) or not raw_common_name.strip():
            raise ValueError("common_name_overrides values must be non-empty strings")
        normalized[raw_title.strip().lower()] = _strip_pdf_extension(raw_common_name)
    return normalized


def _infer_common_paper_name(
    entry_line: str | None,
    paper_id: str,
    config,
    explicit_file_name: str | None = None,
) -> str:
    if explicit_file_name is not None and explicit_file_name.strip():
        return _strip_pdf_extension(explicit_file_name)

    if entry_line is not None:
        alias = _extract_common_name_alias_from_entry_line(entry_line)
        if alias:
            return alias

        title = _extract_paper_title_from_entry_line(entry_line)
        if title:
            override = config["common_name_overrides"].get(title.lower())
            if override:
                return override

    return paper_id


def _build_remote_name(
    note_path: Path,
    paper_id: str,
    config,
    entry_line: str | None = None,
    explicit_file_name: str | None = None,
) -> str:
    note_name = _sanitize_path_component(note_path.stem)
    paper_name_source = _infer_common_paper_name(
        entry_line=entry_line,
        paper_id=paper_id,
        config=config,
        explicit_file_name=explicit_file_name,
    )
    paper_id_name = _sanitize_path_component(paper_id)
    if config["paper_name_policy"] == "note-name" and note_path.stem.strip():
        paper_name_source = note_path.stem
    elif config["paper_name_policy"] == "paper-id":
        paper_name_source = paper_id
    paper_name = _sanitize_path_component(paper_name_source)
    return config["remote_path_template"].format(
        note_name=note_name,
        paper_id=paper_id_name,
        paper_name=paper_name,
    )


def _get_default_config_candidates():
    return [DEFAULT_CONFIG_PATH, USER_CONFIG_PATH]


def _resolve_config_path(config_path=None) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser()

    for candidate in _get_default_config_candidates():
        if candidate.exists():
            return candidate

    return DEFAULT_CONFIG_PATH


def load_runtime_config(config_path=None):
    resolved_config_path = _resolve_config_path(config_path)
    if not resolved_config_path.exists():
        searched_paths = ", ".join(
            str(path) for path in _get_default_config_candidates()
        )
        raise FileNotFoundError(
            "Config file not found. Checked explicit/default paths: "
            f"{resolved_config_path if config_path is not None else searched_paths}"
        )

    try:
        raw_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Config file is not valid JSON: {resolved_config_path}"
        ) from error

    if not isinstance(raw_config, dict):
        raise ValueError("Config JSON must be an object")

    drive_folder_id = (raw_config.get("drive_folder_id") or "").strip()
    if not drive_folder_id:
        raise ValueError("Config must define a non-empty drive_folder_id")

    link_type = raw_config.get("link_type", DEFAULT_LINK_TYPE)
    if link_type not in ALLOWED_LINK_TYPES:
        raise ValueError(
            f"Invalid link_type {link_type!r}; expected one of {sorted(ALLOWED_LINK_TYPES)!r}"
        )

    paper_name_policy = raw_config.get("paper_name_policy", DEFAULT_PAPER_NAME_POLICY)
    if paper_name_policy not in ALLOWED_PAPER_NAME_POLICIES:
        raise ValueError(
            "Invalid paper_name_policy "
            f"{paper_name_policy!r}; expected one of {sorted(ALLOWED_PAPER_NAME_POLICIES)!r}"
        )

    remote_path_template = raw_config.get(
        "remote_path_template", DEFAULT_REMOTE_PATH_TEMPLATE
    )
    if not isinstance(remote_path_template, str) or not remote_path_template.strip():
        raise ValueError("remote_path_template must be a non-empty string")
    if (
        paper_name_policy == LEGACY_DEFAULT_PAPER_NAME_POLICY
        and remote_path_template == LEGACY_DEFAULT_REMOTE_PATH_TEMPLATE
    ):
        paper_name_policy = DEFAULT_PAPER_NAME_POLICY
        remote_path_template = DEFAULT_REMOTE_PATH_TEMPLATE
    _validate_template_placeholders(remote_path_template)

    common_name_overrides = _normalize_common_name_overrides(
        raw_config.get("common_name_overrides")
    )

    client_secret_file = Path((raw_config.get("client_secret_file") or "")).expanduser()
    token_file = Path((raw_config.get("token_file") or "")).expanduser()
    if not raw_config.get("client_secret_file"):
        raise ValueError("Config must define client_secret_file")
    if not raw_config.get("token_file"):
        raise ValueError("Config must define token_file")
    if not client_secret_file.exists():
        raise FileNotFoundError(
            f"Google Drive client_secret_file does not exist: {client_secret_file}"
        )

    return {
        "config_path": resolved_config_path,
        "drive_folder_id": drive_folder_id,
        "link_type": link_type,
        "paper_name_policy": paper_name_policy,
        "remote_path_template": remote_path_template,
        "common_name_overrides": common_name_overrides,
        "client_secret_file": client_secret_file,
        "token_file": token_file,
    }


def _authorize_interactively(client_secret_file: Path, scopes, token_file: Path):
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise GoogleDriveUploadError(
            "Google Drive OAuth browser flow dependency is not installed"
        ) from error

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(client_secret_file), scopes
        )
        credentials = flow.run_local_server(port=0, open_browser=True)
    except Exception as error:
        raise GoogleDriveUploadError(
            "Failed to complete Google Drive browser authorization flow"
        ) from error

    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(credentials.to_json(), encoding="utf-8")
    return credentials


def find_paper_entry_line(note_text: str, paper_identifier: str) -> str:
    # Collect all bullet lines that mention the paper identifier
    matches = []
    for line in note_text.splitlines():
        if paper_identifier in line and line.lstrip().startswith("- "):
            matches.append(line)
    if len(matches) == 0:
        raise ValueError(f"Could not find paper entry for: {paper_identifier}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple paper entries found for {paper_identifier}: {matches!r}"
        )
    return matches[0]


def _find_candidate_paper_entry_lines(note_text: str):
    return [
        line
        for line in note_text.splitlines()
        if line.lstrip().startswith("- **") and "](http" in line
    ]


def _infer_paper_id_from_note(note_text: str) -> str:
    candidate_lines = _find_candidate_paper_entry_lines(note_text)
    if len(candidate_lines) != 1:
        raise ValueError(
            "paper_id is required when the note contains zero or multiple paper entries"
        )

    match = re.match(r"-\s+\*\*(.+?)\*\*", candidate_lines[0].lstrip())
    if not match:
        raise ValueError("Could not infer paper_id from the unambiguous citation line")
    return match.group(1)


def replace_or_append_my_pdf(line: str, new_url: str) -> str:
    pattern = re.compile(r"\(?\[My PDF\]\([^\)]+\)\)?")
    matches = list(pattern.finditer(line))
    if not matches:
        return f"{line} ([My PDF]({new_url}))"
    new_line = line
    for m in reversed(matches[1:]):
        new_line = new_line[: m.start()] + new_line[m.end() :]
    first = matches[0]
    new_line = (
        new_line[: first.start()] + f"([My PDF]({new_url}))" + new_line[first.end() :]
    )
    return new_line


def _replace_my_pdf_url(line: str, new_url: str) -> str:
    pattern = re.compile(r"\[My PDF\]\([^)]+\)")
    matches = list(pattern.finditer(line))
    if not matches:
        raise ValueError("Line does not contain a My PDF link")
    new_line = line
    for match in reversed(matches[1:]):
        new_line = new_line[: match.start()] + new_line[match.end() :]
    first = matches[0]
    return new_line[: first.start()] + f"[My PDF]({new_url})" + new_line[first.end() :]


def _extract_drive_file_id_from_my_pdf(line: str) -> str | None:
    match = re.search(
        r"\[My PDF\]\(https://drive\.google\.com/file/d/([^/)]+)/[^)]*\)", line
    )
    if not match:
        return None
    return match.group(1)


def _iter_markdown_files(note_root: Path):
    if note_root.is_file():
        if note_root.suffix.lower() == ".md":
            yield note_root
        return

    for path in sorted(note_root.rglob("*.md")):
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        yield path


def _find_existing_drive_pdf_entries(note_path: Path, config):
    note_text = note_path.read_text(encoding="utf-8")
    entries = []
    for line_number, line in enumerate(note_text.splitlines(), start=1):
        file_id = _extract_drive_file_id_from_my_pdf(line)
        if not file_id:
            continue
        paper_title = _extract_paper_title_from_entry_line(line)
        if not paper_title:
            continue
        remote_name = _build_remote_name(
            note_path=note_path,
            paper_id=paper_title,
            config=config,
            entry_line=line,
        )
        entries.append(
            {
                "note_path": note_path,
                "line_number": line_number,
                "line": line,
                "file_id": file_id,
                "paper_id": paper_title,
                "remote_name": remote_name,
            }
        )
    return entries


def extract_download_url(line: str) -> str:
    link_matches = re.findall(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", line)
    candidate_links = [
        (label.strip(), url)
        for label, url in link_matches
        if label.strip().lower() != "my pdf"
    ]

    for _, url in candidate_links:
        if re.search(r"\.pdf(?:$|[?#])", url, flags=re.IGNORECASE):
            return url

    for _, url in candidate_links:
        arxiv_match = re.match(
            r"https?://arxiv\.org/abs/([0-9]+\.[0-9]+)(?:v\d+)?(?:[?#].*)?$",
            url,
            flags=re.IGNORECASE,
        )
        if arxiv_match:
            paper_id = arxiv_match.group(1)
            return f"https://arxiv.org/pdf/{paper_id}.pdf"

    for _, url in candidate_links:
        parsed_url = urlparse(url)
        if parsed_url.path.lower().endswith("/pdf"):
            return url

    raise ValueError("Could not resolve a downloadable paper URL from citation line")


def download_pdf_to_temp(download_url: str, filename_hint: str) -> Path:
    """Download a PDF to a new temp directory.

    The caller owns cleanup of the returned file path and its parent temp directory.
    This version validates filename hints and ensures cleanup of temp artifacts on
    any failure after the temp dir is created.
    """

    temp_dir = None
    pdf_path = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="paper-upload-"))
        if not filename_hint or not filename_hint.strip():
            raise ValueError("filename_hint must not be empty")
        if Path(filename_hint).is_absolute():
            raise ValueError("filename_hint must not be an absolute path")
        if "/" in filename_hint or "\\" in filename_hint:
            raise ValueError("filename_hint must not contain path separators")
        if filename_hint in {".", ".."}:
            raise ValueError("filename_hint must not be '.' or '..'")

        pdf_path = temp_dir / f"{filename_hint}.pdf"
        request = urllib.request.Request(download_url)

        with urllib.request.urlopen(request, timeout=30) as response:
            status_code = response.getcode()
            if status_code is not None and status_code >= 400:
                raise ValueError(f"Failed to download PDF: HTTP {status_code}")

            content_type = (response.headers.get("Content-Type") or "").lower()
            first_chunk = response.read(8192)
            if not first_chunk:
                raise ValueError("Downloaded file is empty")

            first_chunk_stripped = first_chunk.lstrip()
            looks_like_pdf = first_chunk_stripped.startswith(b"%PDF-")
            looks_like_html = first_chunk_stripped.startswith(
                (b"<!doctype html", b"<html")
            )

            if "text/html" in content_type or "application/xhtml+xml" in content_type:
                raise ValueError("Download URL returned HTML instead of PDF")
            if looks_like_html:
                raise ValueError("Download URL appears to return an HTML page")
            if not looks_like_pdf and "application/pdf" not in content_type:
                raise ValueError("Downloaded content does not appear to be a PDF")

            with pdf_path.open("wb") as output_file:
                output_file.write(first_chunk)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output_file.write(chunk)

        return pdf_path
    except Exception:
        # Cleanup any created temporary artifacts on failure, then re-raise
        import shutil

        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(str(temp_dir), ignore_errors=True)
        raise


def upload_pdf_and_get_link(drive_client, pdf_path: Path, remote_name: str) -> str:
    file_id = drive_client.upload_pdf(pdf_path, remote_name)
    return drive_client.get_share_link(file_id)


class GoogleDriveClient:
    def __init__(
        self, service, drive_folder_id: str, link_type: str = DEFAULT_LINK_TYPE
    ):
        self.service = service
        self.drive_folder_id = drive_folder_id
        self.link_type = link_type

    def upload_pdf(self, pdf_path: Path, remote_name: str) -> str:
        if not self.drive_folder_id:
            raise ValueError("drive_folder_id must not be empty")
        if not remote_name or not remote_name.strip():
            raise ValueError("remote_name must not be empty")
        if not pdf_path.exists():
            raise GoogleDriveUploadError(f"PDF file does not exist: {pdf_path}")

        try:
            from googleapiclient.http import MediaFileUpload  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise GoogleDriveUploadError(
                "Google Drive client dependencies are not installed"
            ) from error

        media = MediaFileUpload(
            str(pdf_path), mimetype="application/pdf", resumable=False
        )
        query = (
            f"name = '{_escape_drive_query_value(remote_name)}' and "
            f"'{_escape_drive_query_value(self.drive_folder_id)}' in parents and "
            "trashed = false"
        )

        try:
            existing_files = []
            page_token = None
            while True:
                response = (
                    self.service.files()
                    .list(
                        q=query,
                        spaces="drive",
                        fields="nextPageToken, files(id, createdTime, modifiedTime)",
                        pageSize=1000,
                        pageToken=page_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )
                existing_files.extend(response.get("files", []))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break

            if existing_files:
                existing_file = _pick_existing_drive_file(existing_files)
                file_id = existing_file["id"] if existing_file else None
                if not file_id:
                    raise GoogleDriveUploadError(
                        f"Google Drive lookup returned no file id for '{remote_name}'"
                    )
                result = (
                    self.service.files()
                    .update(
                        fileId=file_id,
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
            else:
                result = (
                    self.service.files()
                    .create(
                        body={
                            "name": remote_name,
                            "parents": [self.drive_folder_id],
                        },
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
        except Exception as error:
            status = _get_http_status(error)
            if status in {401, 403}:
                raise GoogleDriveUploadError(
                    f"Google Drive upload permission failed for folder {self.drive_folder_id}"
                ) from error
            raise GoogleDriveUploadError(
                f"Google Drive upload failed for '{remote_name}'"
            ) from error

        file_id = result.get("id")
        if not file_id:
            raise GoogleDriveUploadError(
                f"Google Drive upload did not return a file id for '{remote_name}'"
            )
        return file_id

    def rename_pdf_and_get_link(self, file_id: str, remote_name: str) -> str:
        if not file_id:
            raise ValueError("file_id must not be empty")
        if not remote_name or not remote_name.strip():
            raise ValueError("remote_name must not be empty")

        try:
            result = (
                self.service.files()
                .update(
                    fileId=file_id,
                    body={"name": remote_name},
                    fields="id",
                    supportsAllDrives=True,
                )
                .execute()
            )
        except Exception as error:
            status = _get_http_status(error)
            if status in {401, 403}:
                raise GoogleDriveUploadError(
                    f"Google Drive rename permission failed for file {file_id}"
                ) from error
            raise GoogleDriveUploadError(
                f"Google Drive rename failed for file {file_id} to '{remote_name}'"
            ) from error

        renamed_file_id = result.get("id") or file_id
        return self.get_share_link(renamed_file_id)

    def get_share_link(self, file_id: str) -> str:
        if not file_id:
            raise ValueError("file_id must not be empty")

        try:
            permissions_response = (
                self.service.permissions()
                .list(
                    fileId=file_id,
                    fields="permissions(id, type, role)",
                    supportsAllDrives=True,
                )
                .execute()
            )
            existing_permissions = permissions_response.get("permissions", [])
            already_shareable = any(
                permission.get("type") == "anyone"
                and permission.get("role") in {"reader", "commenter", "writer"}
                for permission in existing_permissions
            )
            if not already_shareable:
                (
                    self.service.permissions()
                    .create(
                        fileId=file_id,
                        body={"role": "reader", "type": "anyone"},
                        fields="id",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
        except Exception as error:
            raise GoogleDrivePermissionError(
                f"Failed to make Google Drive file '{file_id}' shareable"
            ) from error

        try:
            file_response = (
                self.service.files()
                .get(
                    fileId=file_id,
                    fields=self.link_type,
                    supportsAllDrives=True,
                )
                .execute()
            )
        except Exception as error:
            raise GoogleDrivePermissionError(
                f"Failed to fetch share link for Google Drive file '{file_id}'"
            ) from error

        share_link = file_response.get(self.link_type)
        if not share_link:
            raise GoogleDrivePermissionError(
                f"Google Drive file '{file_id}' did not return {self.link_type}"
            )
        return share_link


def create_drive_client_from_config(config):
    try:
        from google.auth.transport.requests import Request  # pyright: ignore[reportMissingImports]
        from google.oauth2.credentials import Credentials  # pyright: ignore[reportMissingImports]
        from googleapiclient.discovery import build  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise GoogleDriveUploadError(
            "Google Drive client dependencies are not installed"
        ) from error

    scopes = ["https://www.googleapis.com/auth/drive"]
    token_file = config["token_file"]
    client_secret_file = config["client_secret_file"]
    if token_file.exists():
        try:
            credentials = Credentials.from_authorized_user_file(str(token_file), scopes)
        except Exception as error:
            raise GoogleDriveUploadError(
                f"Failed to load Google Drive token file: {token_file}"
            ) from error
    else:
        credentials = _authorize_interactively(client_secret_file, scopes, token_file)

    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except Exception as error:
                raise GoogleDriveUploadError(
                    "Google Drive credentials are expired and could not be refreshed"
                ) from error
            token_file.write_text(credentials.to_json(), encoding="utf-8")
        else:
            credentials = _authorize_interactively(
                client_secret_file, scopes, token_file
            )

    try:
        service = build("drive", "v3", credentials=credentials)
    except Exception as error:
        raise GoogleDriveUploadError(
            "Failed to create Google Drive service client"
        ) from error

    return GoogleDriveClient(
        service=service,
        drive_folder_id=config["drive_folder_id"],
        link_type=config["link_type"],
    )


def process_note_entry(
    note_path,
    paper_id: str | None,
    drive_client,
    remote_name: str,
    downloader=download_pdf_to_temp,
    dry_run: bool = False,
) -> str:
    resolved_note_path = Path(note_path)
    note_text = resolved_note_path.read_text(encoding="utf-8")
    if paper_id is None:
        paper_id = _infer_paper_id_from_note(note_text)
    entry_line = find_paper_entry_line(note_text, paper_id)
    download_url = extract_download_url(entry_line)
    filename_hint = Path(remote_name).stem or _sanitize_path_component(paper_id)

    temp_pdf_path = None
    try:
        temp_pdf_path = downloader(download_url, filename_hint)
        share_link = upload_pdf_and_get_link(drive_client, temp_pdf_path, remote_name)
        if not dry_run:
            patched_line = replace_or_append_my_pdf(entry_line, share_link)
            patched_text = note_text.replace(entry_line, patched_line, 1)
            resolved_note_path.write_text(patched_text, encoding="utf-8")
        return share_link
    finally:
        if temp_pdf_path is not None:
            shutil.rmtree(str(temp_pdf_path.parent), ignore_errors=True)


def rename_existing_drive_pdfs(note_root, drive_client, config, dry_run: bool = False):
    resolved_note_root = Path(note_root)
    entries_by_note = {}
    for note_path in _iter_markdown_files(resolved_note_root):
        entries = _find_existing_drive_pdf_entries(note_path, config)
        if entries:
            entries_by_note[note_path] = entries

    operations = []
    patched_lines = {}
    for note_path, entries in entries_by_note.items():
        patched_lines[note_path] = []
        for entry in entries:
            if dry_run:
                share_link = f"https://drive.google.com/file/d/{entry['file_id']}/view"
            else:
                share_link = drive_client.rename_pdf_and_get_link(
                    entry["file_id"], entry["remote_name"]
                )
            patched_line = _replace_my_pdf_url(entry["line"], share_link)
            patched_lines[note_path].append((entry["line"], patched_line))
            operations.append(
                {
                    "note_path": str(note_path),
                    "line_number": entry["line_number"],
                    "paper_id": entry["paper_id"],
                    "file_id": entry["file_id"],
                    "remote_name": entry["remote_name"],
                    "share_link": share_link,
                }
            )

    if dry_run:
        return operations

    for note_path, replacements in patched_lines.items():
        note_text = note_path.read_text(encoding="utf-8")
        patched_text = note_text
        for old_line, new_line in replacements:
            patched_text = patched_text.replace(old_line, new_line, 1)
        note_path.write_text(patched_text, encoding="utf-8")

    return operations


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Upload a paper PDF to Google Drive")
    parser.add_argument("--note-path", required=True, help="Path to the note to update")
    parser.add_argument(
        "--paper-id",
        help="Paper identifier used to select the citation line; optional when the note is unambiguous",
    )
    parser.add_argument(
        "--file-name",
        help=(
            "Optional common PDF file name to upload as, without any folder path. "
            "The .pdf suffix is optional. Examples: ATSS, transformer"
        ),
    )
    parser.add_argument(
        "--config-path",
        help=(
            "Optional path to config.json; defaults to auto-discovery in the skill "
            "directory and ~/.config/google-drive-paper-upload/config.json"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run upload orchestration without writing the note back to disk",
    )
    parser.add_argument(
        "--rename-existing",
        action="store_true",
        help="Rename existing Google Drive My PDF files and update their Markdown links",
    )
    parser.add_argument(
        "--note-root",
        help="Markdown file or directory to scan when --rename-existing is used",
    )
    return parser


def main(argv=None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        config = load_runtime_config(args.config_path)
        if args.rename_existing:
            note_root = Path(args.note_root or args.note_path)
            drive_client = None if args.dry_run else create_drive_client_from_config(config)
            operations = rename_existing_drive_pdfs(
                note_root=note_root,
                drive_client=drive_client,
                config=config,
                dry_run=args.dry_run,
            )
            print(json.dumps(operations, ensure_ascii=False, indent=2))
            return 0

        note_path = Path(args.note_path)
        paper_id = args.paper_id
        note_text = note_path.read_text(encoding="utf-8")
        if paper_id is None:
            paper_id = _infer_paper_id_from_note(note_text)
        entry_line = find_paper_entry_line(note_text, paper_id)
        remote_name = _build_remote_name(
            note_path=note_path,
            paper_id=paper_id,
            config=config,
            entry_line=entry_line,
            explicit_file_name=args.file_name,
        )
        drive_client = create_drive_client_from_config(config)
        share_link = process_note_entry(
            note_path=note_path,
            paper_id=paper_id,
            drive_client=drive_client,
            remote_name=remote_name,
            dry_run=args.dry_run,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(share_link)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
