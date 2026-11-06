"""Unit tests for workspace helpers (issue IDs, sandbox creation)."""

from aidevteam.workspace import create_workspace, workspace_for


def test_create_workspace_uses_issue_id_as_directory_name(tmp_path):
    workspace = create_workspace(12345, root=tmp_path)
    assert workspace == tmp_path / "12345"
    assert workspace.is_dir()


def test_create_workspace_is_idempotent(tmp_path):
    """Revision iterations re-derive the same workspace; creating twice is fine."""
    first = create_workspace(12345, root=tmp_path)
    second = create_workspace(12345, root=tmp_path)
    assert first == second == tmp_path / "12345"


def test_workspace_for_derives_path_without_creating(tmp_path):
    assert workspace_for(42, root=tmp_path) == tmp_path / "42"
    assert not (tmp_path / "42").exists()  # derivation never touches the disk