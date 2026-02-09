"""Persistence layer for conversations and tasks with thread-safe file operations."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4


class ConversationStore:
    """Persists conversations and task snapshots to disk with asyncio-safe operations."""

    MAX_CONVERSATIONS = 50
    """Maximum number of conversations to keep before pruning oldest."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the conversation store.

        Args:
            data_dir: Directory to store data. Defaults to ~/.codehorde/
        """
        if data_dir is None:
            self._dir = Path.home() / ".codehorde"
        else:
            self._dir = Path(data_dir)

        # Create directory if it doesn't exist
        self._dir.mkdir(parents=True, exist_ok=True)

        # Asyncio lock for thread-safe file operations
        self._lock = asyncio.Lock()

        # Load existing data into memory
        self._data = self._load()

    def _data_path(self) -> Path:
        """Get the path to the data.json file."""
        return self._dir / "data.json"

    def _load(self) -> dict:
        """
        Load data from disk.

        Returns:
            Dictionary with conversations and tasks. Returns default structure if file is missing.
        """
        path = self._data_path()

        if not path.exists():
            return {
                "conversations": [],
                "active_conversation_id": None,
                "tasks": []
            }

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Ensure structure has all required keys
                data.setdefault("conversations", [])
                data.setdefault("active_conversation_id", None)
                data.setdefault("tasks", [])
                return data
        except (json.JSONDecodeError, IOError) as e:
            # If file is corrupted, return default structure
            print(f"Warning: Failed to load data.json: {e}. Starting with empty store.")
            return {
                "conversations": [],
                "active_conversation_id": None,
                "tasks": []
            }

    async def _save(self) -> None:
        """
        Save current state to disk with asyncio lock.

        This method is thread-safe and serializes access to the file.
        """
        async with self._lock:
            path = self._data_path()
            try:
                # Write to temporary file first, then rename for atomicity
                temp_path = path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
                temp_path.replace(path)
            except IOError as e:
                print(f"Error: Failed to save data.json: {e}")

    # ── Conversations ──────────────────────────────────────

    async def list_conversations(self) -> list[dict]:
        """
        Return all conversations with metadata.

        Returns:
            List of conversation dicts with {id, title, created_at, updated_at, message_count}.
        """
        return [
            {
                "id": conv["id"],
                "title": conv["title"],
                "created_at": conv["created_at"],
                "updated_at": conv["updated_at"],
                "message_count": len(conv.get("messages", []))
            }
            for conv in self._data["conversations"]
        ]

    async def get_conversation(self, conv_id: str) -> Optional[dict]:
        """
        Get a full conversation with all messages.

        Args:
            conv_id: The conversation ID.

        Returns:
            Full conversation dict or None if not found.
        """
        for conv in self._data["conversations"]:
            if conv["id"] == conv_id:
                return conv
        return None

    async def create_conversation(self, title: str = "New Chat") -> dict:
        """
        Create a new empty conversation.

        Args:
            title: Optional conversation title. Defaults to "New Chat".

        Returns:
            New conversation dict with {id, title, created_at, messages: []}.
        """
        now = datetime.utcnow().isoformat() + "Z"
        conv = {
            "id": str(uuid4()),
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": []
        }
        self._data["conversations"].append(conv)
        await self._save()
        return conv

    async def add_message(
        self,
        conv_id: str,
        role: str,
        text: str,
        agent: Optional[str] = None
    ) -> Optional[dict]:
        """
        Add a message to a conversation.

        Auto-generates conversation title from first user message if title is still "New Chat".

        Args:
            conv_id: The conversation ID.
            role: Message role ("user" or "assistant").
            text: Message text.
            agent: Optional agent identifier (for assistant messages).

        Returns:
            The message dict or None if conversation not found.
        """
        conv = await self.get_conversation(conv_id)
        if conv is None:
            return None

        now = datetime.utcnow().isoformat() + "Z"
        message = {
            "role": role,
            "text": text,
            "timestamp": now
        }

        # Add agent identifier if provided
        if agent is not None:
            message["agent"] = agent

        conv["messages"].append(message)
        conv["updated_at"] = now

        # Auto-generate title from first user message
        if conv["title"] == "New Chat" and role == "user":
            title_preview = text[:50].strip()
            # Remove newlines and truncate at sentence boundary if possible
            title_preview = title_preview.split('\n')[0]
            if len(text) > 50:
                title_preview += "..."
            conv["title"] = title_preview

        await self._save()
        return message

    async def delete_conversation(self, conv_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conv_id: The conversation ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        initial_count = len(self._data["conversations"])
        self._data["conversations"] = [
            conv for conv in self._data["conversations"]
            if conv["id"] != conv_id
        ]

        if len(self._data["conversations"]) < initial_count:
            # Clear active conversation if it was deleted
            if self._data.get("active_conversation_id") == conv_id:
                self._data["active_conversation_id"] = None
            await self._save()
            return True

        return False

    async def rename_conversation(self, conv_id: str, title: str) -> bool:
        """
        Rename a conversation.

        Args:
            conv_id: The conversation ID.
            title: New conversation title.

        Returns:
            True if renamed, False if conversation not found.
        """
        conv = await self.get_conversation(conv_id)
        if conv is None:
            return False

        conv["title"] = title
        await self._save()
        return True

    # ── Tasks persistence ──────────────────────────────────

    async def save_tasks(self, tasks: list[dict]) -> None:
        """
        Snapshot the current task list to disk.

        Args:
            tasks: List of task dictionaries to save.
        """
        self._data["tasks"] = tasks
        await self._save()

    async def load_tasks(self) -> list[dict]:
        """
        Load previously saved tasks.

        Returns:
            List of saved task dictionaries.
        """
        return self._data.get("tasks", [])

    # ── Utility methods ────────────────────────────────────

    async def set_active_conversation(self, conv_id: Optional[str]) -> None:
        """
        Set the active conversation ID.

        Args:
            conv_id: The conversation ID to set as active, or None.
        """
        self._data["active_conversation_id"] = conv_id
        await self._save()

    async def get_active_conversation_id(self) -> Optional[str]:
        """
        Get the active conversation ID.

        Returns:
            The active conversation ID or None.
        """
        return self._data.get("active_conversation_id")

    async def _prune_conversations(self) -> None:
        """
        Prune conversations if exceeding MAX_CONVERSATIONS.

        Deletes the oldest conversations until count is at max.
        """
        if len(self._data["conversations"]) > self.MAX_CONVERSATIONS:
            # Sort by created_at and remove oldest
            self._data["conversations"].sort(key=lambda c: c["created_at"])
            to_remove = len(self._data["conversations"]) - self.MAX_CONVERSATIONS
            self._data["conversations"] = self._data["conversations"][to_remove:]
            await self._save()

    async def export_conversation(self, conv_id: str) -> Optional[str]:
        """
        Export a conversation as formatted text.

        Args:
            conv_id: The conversation ID to export.

        Returns:
            Formatted conversation text or None if not found.
        """
        conv = await self.get_conversation(conv_id)
        if conv is None:
            return None

        lines = [
            f"# {conv['title']}",
            f"Created: {conv['created_at']}",
            f"Updated: {conv['updated_at']}",
            ""
        ]

        for msg in conv.get("messages", []):
            role = msg["role"].upper()
            agent_info = f" ({msg['agent']})" if msg.get("agent") else ""
            timestamp = msg.get("timestamp", "")
            lines.append(f"[{role}{agent_info}] {timestamp}")
            lines.append(msg["text"])
            lines.append("")

        return "\n".join(lines)

    async def clear_all(self) -> None:
        """
        Clear all conversations and tasks (destructive operation).

        Use with caution.
        """
        self._data = {
            "conversations": [],
            "active_conversation_id": None,
            "tasks": []
        }
        await self._save()

    async def get_stats(self) -> dict:
        """
        Get store statistics.

        Returns:
            Dictionary with conversation count, message count, and tasks count.
        """
        total_messages = sum(
            len(conv.get("messages", []))
            for conv in self._data["conversations"]
        )
        return {
            "conversation_count": len(self._data["conversations"]),
            "total_messages": total_messages,
            "task_count": len(self._data.get("tasks", [])),
            "storage_dir": str(self._dir)
        }
