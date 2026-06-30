from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from entropyos.models import AuditEntry


class AuditLogger:
    def __init__(self, path: str | Path = ""):
        self.path = Path(path) if path else Path("entropy_audit.log")
        self._buffer: list[AuditEntry] = []
        self._flush_threshold = 100

    def log(
        self,
        action: str,
        user_id: str = "",
        resource: str = "",
        detail: str = "",
        ip: str = "",
    ) -> None:
        entry = AuditEntry(
            action=action,
            user_id=user_id,
            resource=resource,
            detail=detail,
            ip=ip,
        )
        self._buffer.append(entry)
        if len(self._buffer) >= self._flush_threshold:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.path.exists() else "w"
        with open(self.path, mode) as f:
            for entry in self._buffer:
                f.write(json.dumps({
                    "timestamp": entry.timestamp,
                    "action": entry.action,
                    "user_id": entry.user_id,
                    "resource": entry.resource,
                    "detail": entry.detail,
                    "ip": entry.ip,
                }) + "\n")
        self._buffer.clear()

    def query(self, action: str | None = None, user_id: str | None = None, limit: int = 100) -> list[AuditEntry]:
        if not self.path.exists():
            return []
        results: list[AuditEntry] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if action and d["action"] != action:
                    continue
                if user_id and d["user_id"] != user_id:
                    continue
                results.append(AuditEntry(**d))
                if len(results) >= limit:
                    break
        return results

    def __del__(self) -> None:
        if self._buffer:
            self.flush()
