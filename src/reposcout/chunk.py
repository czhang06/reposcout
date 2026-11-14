"""The unit of retrieval shared by ingestion, stores and the agent."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Literal

SourceType = Literal["code", "doc", "issue", "pr"]


@dataclass(frozen=True)
class Chunk:
    source_type: SourceType
    path: str  # repo-relative, forward slashes; "pr/<number>" for pull requests
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str
    symbol: str | None = None  # e.g. "TaskInstance.clear" or a doc heading path
    commit: str | None = None
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def content_hash(self) -> str:
        """Hash of the text alone, used to skip re-embedding unchanged chunks."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def id(self) -> str:
        """Stable id: same location and same text give the same id."""
        key = f"{self.source_type}|{self.path}|{self.start_line}|{self.end_line}|{self.content_hash}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]

    @property
    def citation(self) -> str:
        if self.source_type == "pr":
            return f"PR #{self.extra['pr']}" if self.extra.get("pr") else self.path
        if self.source_type == "issue":
            return self.path
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def embed_text(self) -> str:
        """Text sent to the embedder: a short header gives the model the file and symbol."""
        header = f"{self.source_type}: {self.path}"
        if self.symbol:
            header += f" :: {self.symbol}"
        return f"{header}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)
