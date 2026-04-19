import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


SCRIPT_PATH = Path(__file__).resolve().parent / "scripts" / "upload_paper_to_gdrive.py"


def load_module():
    spec = importlib.util.spec_from_file_location("upload_paper_to_gdrive", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_note_entry_updates_exact_selected_line_and_cleans_temp(tmp_path):
    module = load_module()
    note_path = tmp_path / "note.md"
    original_text = "\n".join(
        [
            "- **Keep Me** ([Paper](https://example.com/keep.pdf))",
            "- **Target Paper** ([Paper](https://arxiv.org/abs/1234.5678))",
        ]
    )
    note_path.write_text(original_text, encoding="utf-8")

    downloaded_pdf = tmp_path / "temp-area" / "target.pdf"
    downloaded_pdf.parent.mkdir()
    downloaded_pdf.write_bytes(b"%PDF-1.7\n")

    class FakeDriveClient:
        def __init__(self):
            self.calls = []

        def upload_pdf(self, pdf_path, remote_name):
            self.calls.append((pdf_path, remote_name))
            return "file-123"

        def get_share_link(self, file_id):
            assert file_id == "file-123"
            return "https://drive.google.com/file/d/file-123/view"

    drive_client = FakeDriveClient()

    result = module.process_note_entry(
        note_path=note_path,
        paper_id="Target Paper",
        drive_client=drive_client,
        remote_name="Target Paper.pdf",
        downloader=lambda download_url, filename_hint: downloaded_pdf,
    )

    assert result == "https://drive.google.com/file/d/file-123/view"
    assert drive_client.calls == [(downloaded_pdf, "Target Paper.pdf")]
    assert downloaded_pdf.parent.exists() is False
    assert note_path.read_text(encoding="utf-8") == "\n".join(
        [
            "- **Keep Me** ([Paper](https://example.com/keep.pdf))",
            "- **Target Paper** ([Paper](https://arxiv.org/abs/1234.5678)) ([My PDF](https://drive.google.com/file/d/file-123/view))",
        ]
    )


def test_replace_or_append_my_pdf_updates_bare_link_in_place():
    module = load_module()
    line = "- **Target Paper** ([Paper](https://example.com/paper.pdf)) [My PDF](https://old.example/view)"

    updated = module.replace_or_append_my_pdf(
        line, "https://drive.google.com/file/d/file-123/view"
    )

    assert updated.count("[My PDF](") == 1
    assert "https://old.example/view" not in updated
    assert "https://drive.google.com/file/d/file-123/view" in updated


def test_load_runtime_config_reads_json_and_expands_paths(tmp_path):
    module = load_module()
    config_path = tmp_path / "config.json"
    token_file = tmp_path / "token.json"
    secret_file = tmp_path / "client_secret.json"
    token_file.write_text("{}", encoding="utf-8")
    secret_file.write_text("{}", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "drive_folder_id": "folder-123",
                "client_secret_file": str(secret_file),
                "token_file": str(token_file),
            }
        ),
        encoding="utf-8",
    )

    config = module.load_runtime_config(config_path)

    assert config["drive_folder_id"] == "folder-123"
    assert config["link_type"] == "webViewLink"
    assert config["paper_name_policy"] == "paper-id"
    assert config["remote_path_template"] == "PaperReading/{note_name}/{paper_id}.pdf"
    assert config["client_secret_file"] == secret_file
    assert config["token_file"] == token_file


def test_load_runtime_config_allows_missing_token_file(tmp_path):
    module = load_module()
    config_path = tmp_path / "config.json"
    secret_file = tmp_path / "client_secret.json"
    token_file = tmp_path / "missing-token.json"
    secret_file.write_text("{}", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "drive_folder_id": "folder-123",
                "client_secret_file": str(secret_file),
                "token_file": str(token_file),
            }
        ),
        encoding="utf-8",
    )

    config = module.load_runtime_config(config_path)

    assert config["token_file"] == token_file
    assert token_file.exists() is False


def test_create_drive_client_from_config_runs_browser_auth_when_token_missing(
    tmp_path, monkeypatch
):
    module = load_module()
    token_file = tmp_path / "oauth" / "token.json"
    client_secret_file = tmp_path / "client_secret.json"
    client_secret_file.write_text("{}", encoding="utf-8")
    config = {
        "drive_folder_id": "folder-123",
        "link_type": "webViewLink",
        "client_secret_file": client_secret_file,
        "token_file": token_file,
    }

    calls = {}

    class FakeCredentials:
        valid = True
        expired = False
        refresh_token = None

        def to_json(self):
            return '{"token": "created"}'

    def fake_authorize(client_secret, scopes, token_path):
        calls["authorize"] = (client_secret, tuple(scopes), token_path)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text('{"token": "created"}', encoding="utf-8")
        return FakeCredentials()

    def fake_build(api_name, version, credentials):
        calls["build"] = (api_name, version, credentials)
        return "fake-service"

    monkeypatch.setattr(module, "_authorize_interactively", fake_authorize)
    monkeypatch.setitem(
        sys.modules,
        "google.auth.transport.requests",
        types.SimpleNamespace(Request=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "google.oauth2.credentials",
        types.SimpleNamespace(
            Credentials=type(
                "C",
                (),
                {
                    "from_authorized_user_file": staticmethod(
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("should not load existing token")
                        )
                    )
                },
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "googleapiclient.discovery",
        types.SimpleNamespace(build=fake_build),
    )

    client = module.create_drive_client_from_config(config)

    assert calls["authorize"][0] == client_secret_file
    assert calls["authorize"][2] == token_file
    assert token_file.read_text(encoding="utf-8") == '{"token": "created"}'
    assert client.service == "fake-service"
    assert client.drive_folder_id == "folder-123"


def test_create_drive_client_from_config_reuses_existing_token(tmp_path, monkeypatch):
    module = load_module()
    token_file = tmp_path / "token.json"
    client_secret_file = tmp_path / "client_secret.json"
    token_file.write_text("{}", encoding="utf-8")
    client_secret_file.write_text("{}", encoding="utf-8")
    config = {
        "drive_folder_id": "folder-123",
        "link_type": "webViewLink",
        "client_secret_file": client_secret_file,
        "token_file": token_file,
    }

    calls = {}

    class FakeCredentials:
        valid = True
        expired = False
        refresh_token = None

    class FakeCredentialsClass:
        @staticmethod
        def from_authorized_user_file(path, scopes):
            calls["from_file"] = (path, tuple(scopes))
            return FakeCredentials()

    def fake_build(api_name, version, credentials):
        calls["build"] = (api_name, version, credentials)
        return "fake-service"

    def fail_authorize(*args, **kwargs):
        raise AssertionError("interactive auth should not run when token exists")

    monkeypatch.setattr(module, "_authorize_interactively", fail_authorize)
    monkeypatch.setitem(
        sys.modules,
        "google.auth.transport.requests",
        types.SimpleNamespace(Request=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "google.oauth2.credentials",
        types.SimpleNamespace(Credentials=FakeCredentialsClass),
    )
    monkeypatch.setitem(
        sys.modules,
        "googleapiclient.discovery",
        types.SimpleNamespace(build=fake_build),
    )

    client = module.create_drive_client_from_config(config)

    assert calls["from_file"][0] == str(token_file)
    assert client.service == "fake-service"
    assert client.link_type == "webViewLink"


def test_main_uses_explicit_config_and_dry_run_flow(tmp_path, monkeypatch, capsys):
    module = load_module()
    note_path = tmp_path / "paper-note.md"
    note_path.write_text(
        "- **Target Paper** ([Paper](https://example.com/paper.pdf))\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    token_file = tmp_path / "token.json"
    secret_file = tmp_path / "client_secret.json"
    token_file.write_text("{}", encoding="utf-8")
    secret_file.write_text("{}", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "drive_folder_id": "folder-123",
                "client_secret_file": str(secret_file),
                "token_file": str(token_file),
            }
        ),
        encoding="utf-8",
    )

    calls = {}

    def fake_create_drive_client(config):
        calls["config"] = config
        return object()

    def fake_process_note_entry(**kwargs):
        calls["process"] = kwargs
        return "https://drive.google.com/file/d/dry-run/view"

    monkeypatch.setattr(
        module, "create_drive_client_from_config", fake_create_drive_client
    )
    monkeypatch.setattr(module, "process_note_entry", fake_process_note_entry)

    exit_code = module.main(
        [
            "--note-path",
            str(note_path),
            "--paper-id",
            "Target Paper",
            "--config-path",
            str(config_path),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "https://drive.google.com/file/d/dry-run/view" in captured.out
    assert calls["config"]["drive_folder_id"] == "folder-123"
    assert calls["process"]["note_path"] == note_path
    assert calls["process"]["paper_id"] == "Target Paper"
    assert calls["process"]["remote_name"] == "PaperReading/paper-note/Target Paper.pdf"
    assert calls["process"]["dry_run"] is True


def test_main_allows_missing_paper_id_for_unambiguous_note(
    tmp_path, monkeypatch, capsys
):
    module = load_module()
    note_path = tmp_path / "single-paper-note.md"
    note_path.write_text(
        "- **Only Paper** ([Paper](https://example.com/paper.pdf))\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    token_file = tmp_path / "token.json"
    secret_file = tmp_path / "client_secret.json"
    token_file.write_text("{}", encoding="utf-8")
    secret_file.write_text("{}", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "drive_folder_id": "folder-123",
                "client_secret_file": str(secret_file),
                "token_file": str(token_file),
            }
        ),
        encoding="utf-8",
    )

    calls = {}

    def fake_create_drive_client(config):
        calls["config"] = config
        return object()

    def fake_process_note_entry(**kwargs):
        calls["process"] = kwargs
        return "https://drive.google.com/file/d/dry-run/view"

    monkeypatch.setattr(
        module, "create_drive_client_from_config", fake_create_drive_client
    )
    monkeypatch.setattr(module, "process_note_entry", fake_process_note_entry)

    exit_code = module.main(
        [
            "--note-path",
            str(note_path),
            "--config-path",
            str(config_path),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "https://drive.google.com/file/d/dry-run/view" in captured.out
    assert calls["process"]["paper_id"] == "Only Paper"
    assert (
        calls["process"]["remote_name"]
        == "PaperReading/single-paper-note/Only Paper.pdf"
    )
