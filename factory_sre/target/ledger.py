"""Crash-reconcilable SQLite command ledger."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from ..contracts import OperationStatus


@dataclass(frozen=True, slots=True)
class Operation:
    idempotency_key: str
    mission_id: str
    asset_id: str
    link_epoch: int
    sequence: int
    command: str
    status: OperationStatus
    response: dict[str, object] | None


class LedgerUnavailable(RuntimeError):
    pass


class OperationLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

    def open(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, check_same_thread=False, timeout=3.0)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            result = connection.execute("PRAGMA integrity_check").fetchone()
            if result != ("ok",):
                raise LedgerUnavailable(f"ledger integrity check failed: {result!r}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS operations (
                    idempotency_key TEXT PRIMARY KEY,
                    mission_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    link_epoch INTEGER NOT NULL,
                    sequence INTEGER NOT NULL,
                    command TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('PENDING','SUCCEEDED','FAILED','UNKNOWN')),
                    response_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(mission_id, link_epoch, sequence)
                );
                """
            )
            connection.commit()
            self._connection = connection
            self.reconcile_pending()
        except (sqlite3.Error, OSError) as exc:
            self._connection = None
            raise LedgerUnavailable(str(exc)) from exc

    @property
    def available(self) -> bool:
        return self._connection is not None

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise LedgerUnavailable("ledger is not available")
        return self._connection

    def reconcile_pending(self) -> int:
        with self._lock:
            cursor = self._db().execute(
                "UPDATE operations SET status='UNKNOWN', updated_at=CURRENT_TIMESTAMP WHERE status='PENDING'"
            )
            self._db().commit()
            return cursor.rowcount

    def get(self, idempotency_key: str) -> Operation | None:
        with self._lock:
            row = self._db().execute(
                "SELECT idempotency_key, mission_id, asset_id, link_epoch, sequence, command, status, response_json "
                "FROM operations WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        response = json.loads(row[7]) if row[7] else None
        return Operation(*row[:6], OperationStatus(row[6]), response)

    def begin(self, *, idempotency_key: str, mission_id: str, asset_id: str, link_epoch: int, sequence: int, command: str) -> Operation:
        with self._lock:
            existing = self.get(idempotency_key)
            if existing is not None:
                return existing
            try:
                self._db().execute(
                    "INSERT INTO operations(idempotency_key, mission_id, asset_id, link_epoch, sequence, command, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'PENDING')",
                    (idempotency_key, mission_id, asset_id, link_epoch, sequence, command),
                )
                self._db().commit()
            except sqlite3.Error as exc:
                self._connection = None
                raise LedgerUnavailable(str(exc)) from exc
            operation = self.get(idempotency_key)
            assert operation is not None
            return operation

    def finish(self, idempotency_key: str, status: OperationStatus, response: dict[str, object]) -> Operation:
        if status not in {OperationStatus.SUCCEEDED, OperationStatus.FAILED, OperationStatus.UNKNOWN}:
            raise ValueError(f"invalid terminal status: {status}")
        with self._lock:
            try:
                self._db().execute(
                    "UPDATE operations SET status=?, response_json=?, updated_at=CURRENT_TIMESTAMP WHERE idempotency_key=?",
                    (status.value, json.dumps(response, sort_keys=True), idempotency_key),
                )
                self._db().commit()
            except sqlite3.Error as exc:
                self._connection = None
                raise LedgerUnavailable(str(exc)) from exc
        operation = self.get(idempotency_key)
        assert operation is not None
        return operation

    def count(self, command: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM operations"
        args: tuple[str, ...] = ()
        if command is not None:
            query += " WHERE command=?"
            args = (command,)
        return int(self._db().execute(query, args).fetchone()[0])

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
