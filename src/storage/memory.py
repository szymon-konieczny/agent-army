"""Persistent memory system for cross-session agent knowledge.

Stores typed memories (decisions, learnings, patterns, blockers, preferences,
context) scoped by project area.  Persists to a JSON file in the project
directory so knowledge survives across server restarts.

Inspired by the memory management pattern from OpenCode / agent instruction files.
"""

import json
import os
import pathlib
import time
from typing import Any, Optional
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)

# Valid memory types
MEMORY_TYPES = {
    "decision",    # Architecture/design choices
    "learning",    # Codebase discoveries
    "preference",  # User/project preferences
    "blocker",     # Known issues/bugs
    "context",     # Feature/system info
    "pattern",     # Code conventions
}

# Valid memory scopes
MEMORY_SCOPES = {
    "project",     # Project-wide
    "auth",        # Authentication related
    "api",         # API related
    "ui",          # UI/frontend related
    "testing",     # Testing related
    "deployment",  # Deployment/infra related
    "database",    # Database related
    "security",    # Security related
}


class MemoryEntry:
    """A single memory entry."""

    def __init__(
        self,
        content: str,
        memory_type: str = "context",
        scope: str = "project",
        tags: Optional[list[str]] = None,
        created_at: Optional[str] = None,
        created_by: Optional[str] = None,
        entry_id: Optional[str] = None,
    ):
        self.id = entry_id or f"mem_{int(time.time() * 1000)}"
        self.content = content
        self.memory_type = memory_type if memory_type in MEMORY_TYPES else "context"
        self.scope = scope if scope in MEMORY_SCOPES else "project"
        self.tags = tags or []
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.created_by = created_by or "system"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.memory_type,
            "scope": self.scope,
            "tags": self.tags,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            content=data.get("content", ""),
            memory_type=data.get("type", "context"),
            scope=data.get("scope", "project"),
            tags=data.get("tags", []),
            created_at=data.get("created_at"),
            created_by=data.get("created_by"),
            entry_id=data.get("id"),
        )

    def matches_query(self, query: str) -> bool:
        """Check if this memory matches a search query."""
        q = query.lower()
        return (
            q in self.content.lower()
            or q in self.memory_type.lower()
            or q in self.scope.lower()
            or any(q in tag.lower() for tag in self.tags)
        )


class AgentMemoryStore:
    """Persistent memory store backed by a JSON file in the project directory.

    Memories are stored in `.codehorde/memory.json` inside the project root.
    """

    MEMORY_DIR = ".codehorde"
    MEMORY_FILE = "memory.json"

    def __init__(self, project_dir: Optional[str] = None):
        self._project_dir = project_dir
        self._memories: list[MemoryEntry] = []
        self._loaded = False

    def set_project_dir(self, project_dir: str) -> None:
        """Update the project directory and reload memories."""
        self._project_dir = project_dir
        self._loaded = False
        self._memories = []

    @property
    def _memory_path(self) -> Optional[pathlib.Path]:
        if not self._project_dir:
            return None
        root = pathlib.Path(self._project_dir)
        return root / self.MEMORY_DIR / self.MEMORY_FILE

    def _ensure_loaded(self) -> None:
        """Lazy-load memories from disk."""
        if self._loaded:
            return
        self._loaded = True

        path = self._memory_path
        if not path or not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._memories = [MemoryEntry.from_dict(m) for m in data.get("memories", [])]
        except (json.JSONDecodeError, IOError, KeyError) as exc:
            logger.warning("memory_load_failed", error=str(exc))

    def _save(self) -> None:
        """Persist memories to disk."""
        path = self._memory_path
        if not path:
            return

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "updated_at": datetime.utcnow().isoformat(),
                "memories": [m.to_dict() for m in self._memories],
            }
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except (IOError, PermissionError) as exc:
            logger.warning("memory_save_failed", error=str(exc))

    def remember(
        self,
        content: str,
        memory_type: str = "context",
        scope: str = "project",
        tags: Optional[list[str]] = None,
        created_by: Optional[str] = None,
    ) -> MemoryEntry:
        """Store a new memory."""
        self._ensure_loaded()
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            scope=scope,
            tags=tags,
            created_by=created_by,
        )
        self._memories.append(entry)
        self._save()
        return entry

    def recall(
        self,
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """Search and retrieve memories.

        Args:
            query: Free-text search across content, tags, and metadata.
            memory_type: Filter by memory type (decision, learning, etc.)
            scope: Filter by scope (project, auth, api, etc.)
            limit: Maximum number of results.

        Returns:
            List of matching memory entries, most recent first.
        """
        self._ensure_loaded()

        results = self._memories[:]

        if memory_type:
            results = [m for m in results if m.memory_type == memory_type]
        if scope:
            results = [m for m in results if m.scope == scope]
        if query:
            results = [m for m in results if m.matches_query(query)]

        # Most recent first
        results.sort(key=lambda m: m.created_at, reverse=True)
        return results[:limit]

    def forget(self, entry_id: str) -> bool:
        """Delete a memory by ID."""
        self._ensure_loaded()
        before = len(self._memories)
        self._memories = [m for m in self._memories if m.id != entry_id]
        if len(self._memories) < before:
            self._save()
            return True
        return False

    def get_all(self) -> list[MemoryEntry]:
        """Get all memories."""
        self._ensure_loaded()
        return sorted(self._memories, key=lambda m: m.created_at, reverse=True)

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        self._ensure_loaded()
        type_counts = {}
        scope_counts = {}
        for m in self._memories:
            type_counts[m.memory_type] = type_counts.get(m.memory_type, 0) + 1
            scope_counts[m.scope] = scope_counts.get(m.scope, 0) + 1
        return {
            "total": len(self._memories),
            "by_type": type_counts,
            "by_scope": scope_counts,
        }

    def format_for_agent_context(
        self,
        query: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """Format relevant memories as context for agent system prompts.

        Returns a formatted string ready to inject into the LLM prompt.
        """
        memories = self.recall(query=query, scope=scope, limit=limit)
        if not memories:
            return ""

        lines = ["=== AGENT MEMORY (cross-session knowledge) ==="]
        for m in memories:
            tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
            lines.append(f"[{m.memory_type.upper()}/{m.scope}]{tag_str} {m.content}")
        lines.append("=== END AGENT MEMORY ===")

        return "\n".join(lines)
