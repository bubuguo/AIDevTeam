"""Unit tests for workspace helpers (run ids, sandbox creation)."""

from aidevteam.workspace import create_workspace, new_run_id, workspace_for


def test_new_run_id_format():
    run_id = new_run_id()
    # <14-char timestamp>-<4 hex chars>
    stamp, suffix = run_id.rsplit("-", 1)
    assert len(stamp) == 15 and stamp[8] == "-"  # 20261024-234247
    assert len(suffix) == 4 and int(suffix, 16) >= 0


def test_new_run_id_unique_within_same_second():
    assert len({new_run_id() for _ in range(50)}) == 50


def test_create_workspace_uses_run_id_as_directory_name(tmp_path):
    workspace = create_workspace("20261024-234247-9f3a", root=tmp_path)
    assert workspace == tmp_path / "20261024-234247-9f3a"
    assert workspace.is_dir()


def test_create_workspace_is_idempotent(tmp_path):
    """Revision iterations re-derive the same workspace; creating twice is fine."""
    first = create_workspace("20261024-234247-9f3a", root=tmp_path)
    second = create_workspace("20261024-234247-9f3a", root=tmp_path)
    assert first == second == tmp_path / "20261024-234247-9f3a"


def test_workspace_for_derives_path_without_creating(tmp_path):
    assert workspace_for("r-1", root=tmp_path) == tmp_path / "r-1"
    assert not (tmp_path / "r-1").exists()  # derivation never touches the disk