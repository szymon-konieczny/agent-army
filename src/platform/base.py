"""Abstract base classes for cross-platform adapters.

Each adapter defines a surface that agents / infra code call into.
OS-specific modules (macos.py, windows.py, linux.py) provide the concrete
implementations.  The `capabilities()` method lets agents introspect what
the current OS actually supports so they can adjust their LLM prompts.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Shared data models ────────────────────────────────────────────────

@dataclass
class PlatformCapability:
    """Describes whether a specific feature is available on this OS."""
    name: str
    available: bool
    reason: str = ""  # e.g. "Requires macOS Shortcuts CLI"


@dataclass
class CalendarEvent:
    """Normalised calendar event returned by CalendarAdapter."""
    id: str
    title: str
    start: str  # ISO-8601
    end: str
    calendar_name: str = ""
    description: str = ""
    location: str = ""


# ── DesktopAdapter ────────────────────────────────────────────────────

class DesktopAdapter(ABC):
    """Controls desktop automation — scripts, clipboard, notifications, shortcuts."""

    @abstractmethod
    async def execute_script(self, script: str, *, script_type: str = "auto",
                             args: Optional[List[str]] = None) -> str:
        """Run a script (AppleScript / PowerShell / bash) and return stdout.

        Parameters
        ----------
        script : str
            Script body or path to a script file.
        script_type : str
            "applescript", "powershell", "bash", or "auto" (adapter decides).
        args : list[str] | None
            Positional arguments passed to the script.
        """
        ...

    @abstractmethod
    async def clipboard_get(self) -> str:
        """Return the current system clipboard text."""
        ...

    @abstractmethod
    async def clipboard_set(self, text: str) -> None:
        """Write *text* to the system clipboard."""
        ...

    @abstractmethod
    async def show_notification(self, title: str, body: str) -> None:
        """Display a desktop notification."""
        ...

    @abstractmethod
    async def list_shortcuts(self) -> List[Dict[str, str]]:
        """List available OS-level shortcuts / automation workflows.

        Each dict should have at least ``{"name": "...", "id": "..."}``.
        Returns an empty list if the OS has no shortcut mechanism.
        """
        ...

    @abstractmethod
    async def run_shortcut(self, name: str, *,
                           input_path: Optional[str] = None,
                           output_path: Optional[str] = None) -> str:
        """Execute a named shortcut and return its output (if any)."""
        ...

    @abstractmethod
    async def get_frontmost_app(self) -> str:
        """Return the name (and optionally window title) of the active app."""
        ...

    @abstractmethod
    async def list_open_apps(self) -> List[str]:
        """Return names of all visible/running applications."""
        ...

    @abstractmethod
    def capabilities(self) -> List[PlatformCapability]:
        """Report what this adapter supports on the current OS."""
        ...


# ── CalendarAdapter ───────────────────────────────────────────────────

class CalendarAdapter(ABC):
    """Access the *local* desktop calendar (Calendar.app / Outlook / GNOME)."""

    @abstractmethod
    async def list_calendars(self) -> List[Dict[str, str]]:
        """Return ``[{"name": "...", "description": "..."}]``."""
        ...

    @abstractmethod
    async def get_events(self, *, start: str, end: str,
                         calendar_name: Optional[str] = None) -> List[CalendarEvent]:
        """Fetch events in an ISO-8601 date range."""
        ...

    @abstractmethod
    async def create_event(self, *, calendar_name: str, title: str,
                           start: str, end: str,
                           description: str = "",
                           location: str = "") -> CalendarEvent:
        """Create a new calendar event and return it."""
        ...

    @abstractmethod
    async def delete_event(self, event_id: str, *,
                           calendar_name: Optional[str] = None) -> bool:
        """Delete an event by id/title.  Returns True on success."""
        ...

    @abstractmethod
    def capabilities(self) -> List[PlatformCapability]:
        ...


# ── InfraAdapter ──────────────────────────────────────────────────────

class InfraAdapter(ABC):
    """Start / manage local infrastructure services (Redis, Postgres…)."""

    @abstractmethod
    def start_service(self, service: str) -> bool:
        """Start *service* (e.g. ``"redis"``, ``"postgres"``).  Returns True on success."""
        ...

    @abstractmethod
    def is_service_running(self, service: str) -> bool:
        """Check whether *service* is currently reachable."""
        ...

    @abstractmethod
    def install_service(self, service: str) -> bool:
        """Install *service* via the native package manager.  Returns True on success."""
        ...


# ── PathsAdapter ──────────────────────────────────────────────────────

class PathsAdapter(ABC):
    """OS-specific filesystem paths & webview toolkit identifiers."""

    @property
    @abstractmethod
    def log_dir(self) -> Path:
        """Directory for application logs."""
        ...

    @property
    @abstractmethod
    def config_dir(self) -> Path:
        """Directory for configuration files."""
        ...

    @property
    @abstractmethod
    def temp_dir(self) -> Path:
        """Scratch / temporary directory."""
        ...

    @property
    @abstractmethod
    def webview_gui(self) -> str:
        """pywebview ``gui`` backend name: ``"cocoa"``, ``"edgechromium"``, ``"gtk"``."""
        ...


# ── SubprocessAdapter ─────────────────────────────────────────────────

class SubprocessAdapter(ABC):
    """Smooth over OS differences in subprocess creation & file perms."""

    @abstractmethod
    def create_process_kwargs(self, *, isolated: bool = False) -> Dict[str, Any]:
        """Return extra kwargs to pass to ``subprocess.Popen`` / ``asyncio.create_subprocess_*``.

        When *isolated* is True the subprocess should be detached from the
        parent's process group (``start_new_session`` on POSIX,
        ``CREATE_NEW_PROCESS_GROUP`` on Windows).
        """
        ...

    @abstractmethod
    def chmod_safe(self, path: Path, mode: int) -> None:
        """Set file permissions.  No-op (with debug log) on Windows."""
        ...

    @property
    @abstractmethod
    def shell_executable(self) -> str:
        """Default shell for ``asyncio.create_subprocess_shell``."""
        ...
