"""Persistent checkpoint storage for the LangGraph pipeline.

The graph state (plan, generated code, review verdict) is written to a SQLite
database instead of process memory, so a run survives a process or container
restart. The `resume` and `list` CLI commands read that database back.

All storage access goes through this module: swapping SQLite for another
backend (e.g. Postgres) only touches this file.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from aidevteam.config import DEFAULT_CHECKPOINT_DB, get_settings
from aidevteam.state import msgpack_allowlist

# How long SQLite waits for a competing writer before giving up. `list` may
# open the database while a `run` in another process is still writing to it.
BUSY_TIMEOUT_SECONDS = 30.0


def checkpoint_db_path(path: str | Path | None = None) -> Path:
    """Resolve the SQLite database file, creating its parent directory.

    Defaults to the configured ``checkpoint_db`` setting; a relative path is
    resolved against the current working directory, which keeps the host and
    the container (WORKDIR=/app) on the same file.
    """
    raw = path if path is not None else get_settings().checkpoint_db
    if not str(raw).strip():
        # An empty setting would resolve to the current directory, which
        # SQLite cannot open as a database file.
        raw = DEFAULT_CHECKPOINT_DB
    resolved = Path(str(raw).strip()).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


@contextmanager
def persistent_checkpointer(path: str | Path | None = None) -> Iterator[SqliteSaver]:
    """Open the SQLite-backed checkpointer; the connection closes on exit.

    The connection is built by hand rather than through
    ``SqliteSaver.from_conn_string()`` because that helper accepts no
    ``serde``, and the msgpack allowlist is what lets the pydantic models in
    TeamState round-trip through the database.
    """
    db_path = checkpoint_db_path(path)
    conn = sqlite3.connect(
        db_path, check_same_thread=False, timeout=BUSY_TIMEOUT_SECONDS
    )
    try:
        saver = SqliteSaver(
            conn,
            serde=JsonPlusSerializer(allowed_msgpack_modules=msgpack_allowlist()),
        )
        saver.setup()
        yield saver
    finally:
        conn.close()


def list_threads(saver: SqliteSaver) -> list[tuple[str, str]]:
    """(thread_id, updated_at) for every stored thread, newest first.

    ``updated_at`` is the ISO timestamp LangGraph stores in the checkpoint
    itself; it sorts lexicographically, so no date parsing is needed.

    Only the *latest* checkpoint of each thread is read. The checkpoints table
    grows with every step of every run and each row holds the full state
    (including the generated source code), so scanning all of them just to
    list the runs would get slower with every run that is added.
    """
    with saver.cursor(transaction=False) as cur:
        rows = cur.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()

    threads: list[tuple[str, str]] = []
    for (thread_id,) in rows:
        latest = saver.get_tuple({"configurable": {"thread_id": thread_id}})
        timestamp = str(latest.checkpoint.get("ts", "")) if latest else ""
        threads.append((thread_id, timestamp))
    return sorted(threads, key=lambda item: item[1], reverse=True)
