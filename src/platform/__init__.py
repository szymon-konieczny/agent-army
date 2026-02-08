"""Platform detection and adapter factory.

Usage::

    from src.platform import detect_platform, get_desktop_adapter, get_calendar_adapter
    print(detect_platform())          # "macos" | "windows" | "linux"
    desktop = get_desktop_adapter()    # singleton, lazily loaded
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import (
        CalendarAdapter,
        DesktopAdapter,
        InfraAdapter,
        PathsAdapter,
        SubprocessAdapter,
    )

# ── Platform detection ────────────────────────────────────────────────

_PLATFORM_MAP = {
    "darwin": "macos",
    "win32": "windows",
    "linux": "linux",
}


def detect_platform() -> str:
    """Return a normalised platform name: ``"macos"``, ``"windows"``, or ``"linux"``."""
    return _PLATFORM_MAP.get(sys.platform, "linux")  # default to linux for BSDs etc.


# ── Lazy singleton cache ──────────────────────────────────────────────

_cache: dict[str, object] = {}


def _get_or_create(key: str, factory):
    if key not in _cache:
        _cache[key] = factory()
    return _cache[key]


# ── Adapter factories ────────────────────────────────────────────────

def get_desktop_adapter() -> "DesktopAdapter":
    """Return the ``DesktopAdapter`` for the current OS (cached singleton)."""
    def _factory():
        plat = detect_platform()
        if plat == "macos":
            from .macos import MacDesktopAdapter
            return MacDesktopAdapter()
        elif plat == "windows":
            from .windows import WindowsDesktopAdapter
            return WindowsDesktopAdapter()
        else:
            from .linux import LinuxDesktopAdapter
            return LinuxDesktopAdapter()
    return _get_or_create("desktop", _factory)


def get_calendar_adapter() -> "CalendarAdapter":
    """Return the ``CalendarAdapter`` for the current OS (cached singleton)."""
    def _factory():
        plat = detect_platform()
        if plat == "macos":
            from .macos import MacCalendarAdapter
            return MacCalendarAdapter()
        elif plat == "windows":
            from .windows import WindowsCalendarAdapter
            return WindowsCalendarAdapter()
        else:
            from .linux import LinuxCalendarAdapter
            return LinuxCalendarAdapter()
    return _get_or_create("calendar", _factory)


def get_infra_adapter() -> "InfraAdapter":
    """Return the ``InfraAdapter`` for the current OS (cached singleton)."""
    def _factory():
        plat = detect_platform()
        if plat == "macos":
            from .macos import MacInfraAdapter
            return MacInfraAdapter()
        elif plat == "windows":
            from .windows import WindowsInfraAdapter
            return WindowsInfraAdapter()
        else:
            from .linux import LinuxInfraAdapter
            return LinuxInfraAdapter()
    return _get_or_create("infra", _factory)


def get_paths_adapter() -> "PathsAdapter":
    """Return the ``PathsAdapter`` for the current OS (cached singleton)."""
    def _factory():
        plat = detect_platform()
        if plat == "macos":
            from .macos import MacPathsAdapter
            return MacPathsAdapter()
        elif plat == "windows":
            from .windows import WindowsPathsAdapter
            return WindowsPathsAdapter()
        else:
            from .linux import LinuxPathsAdapter
            return LinuxPathsAdapter()
    return _get_or_create("paths", _factory)


def get_subprocess_adapter() -> "SubprocessAdapter":
    """Return the ``SubprocessAdapter`` for the current OS (cached singleton)."""
    def _factory():
        plat = detect_platform()
        if plat == "macos":
            from .macos import MacSubprocessAdapter
            return MacSubprocessAdapter()
        elif plat == "windows":
            from .windows import WindowsSubprocessAdapter
            return WindowsSubprocessAdapter()
        else:
            from .linux import LinuxSubprocessAdapter
            return LinuxSubprocessAdapter()
    return _get_or_create("subprocess", _factory)
