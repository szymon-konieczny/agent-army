"""Linux adapter implementations.

Provides bash-native desktop automation, xclip/xdg clipboard,
notify-send notifications, systemctl + Docker infrastructure,
and XDG-standard paths.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    CalendarAdapter,
    CalendarEvent,
    DesktopAdapter,
    InfraAdapter,
    PathsAdapter,
    PlatformCapability,
    SubprocessAdapter,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────

def _is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError, socket.timeout):
        return False


def _wait_for_port(port: int, max_wait: float = 15.0) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < max_wait:
        if _is_port_open(port):
            return True
        time.sleep(0.5)
    return False


def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


# ═══════════════════════════════════════════════════════════════════════
# Adapter implementations
# ═══════════════════════════════════════════════════════════════════════

class LinuxDesktopAdapter(DesktopAdapter):
    """Linux desktop automation via bash, xclip, notify-send."""

    async def execute_script(self, script: str, *, script_type: str = "auto",
                             args: Optional[List[str]] = None) -> str:
        if script_type == "applescript":
            raise NotImplementedError(
                "AppleScript is not available on Linux. "
                "Use bash script_type instead."
            )
        # Default to bash on Linux
        proc = await asyncio.create_subprocess_shell(
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode().strip()
            raise RuntimeError(f"Script failed (rc={proc.returncode}): {err}")
        return stdout.decode().strip()

    async def clipboard_get(self) -> str:
        # Try xclip first, then xsel, then wl-paste (Wayland)
        for cmd in [
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "--clipboard", "--output"],
            ["wl-paste"],
        ]:
            if _has_command(cmd[0]):
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0:
                    return stdout.decode().strip()
        raise RuntimeError("No clipboard tool found. Install xclip, xsel, or wl-clipboard.")

    async def clipboard_set(self, text: str) -> None:
        for cmd_name, cmd in [
            ("xclip", ["xclip", "-selection", "clipboard"]),
            ("xsel", ["xsel", "--clipboard", "--input"]),
            ("wl-copy", ["wl-copy"]),
        ]:
            if _has_command(cmd_name):
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate(input=text.encode())
                if proc.returncode == 0:
                    return
        raise RuntimeError("No clipboard tool found. Install xclip, xsel, or wl-clipboard.")

    async def show_notification(self, title: str, body: str) -> None:
        if _has_command("notify-send"):
            proc = await asyncio.create_subprocess_exec(
                "notify-send", title, body,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        else:
            logger.warning("notify-send not found — notification skipped: %s", title)

    async def list_shortcuts(self) -> List[Dict[str, str]]:
        # No native shortcut system on Linux equivalent to macOS Shortcuts
        return []

    async def run_shortcut(self, name: str, *,
                           input_path: Optional[str] = None,
                           output_path: Optional[str] = None) -> str:
        raise NotImplementedError(
            "macOS Shortcuts are not available on Linux. "
            "Use bash scripts or systemd services instead."
        )

    async def get_frontmost_app(self) -> str:
        # xdotool for X11
        if _has_command("xdotool"):
            proc = await asyncio.create_subprocess_exec(
                "xdotool", "getactivewindow", "getwindowname",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                return stdout.decode().strip()
        return "Unknown (install xdotool for active window detection)"

    async def list_open_apps(self) -> List[str]:
        if _has_command("wmctrl"):
            proc = await asyncio.create_subprocess_exec(
                "wmctrl", "-l",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                lines = stdout.decode().strip().splitlines()
                # wmctrl -l format: <id> <desktop> <host> <title>
                return [" ".join(line.split()[3:]) for line in lines if len(line.split()) > 3]
        return []

    def capabilities(self) -> List[PlatformCapability]:
        return [
            PlatformCapability("applescript", False, "Not available on Linux"),
            PlatformCapability("osascript", False, "Not available on Linux"),
            PlatformCapability("bash", True),
            PlatformCapability("shortcuts", False, "macOS Shortcuts not available on Linux"),
            PlatformCapability("clipboard", _has_command("xclip") or _has_command("xsel") or _has_command("wl-paste"),
                               "Via xclip / xsel / wl-clipboard"),
            PlatformCapability("notifications", _has_command("notify-send"), "Via notify-send"),
            PlatformCapability("frontmost_app", _has_command("xdotool"), "Via xdotool"),
            PlatformCapability("list_open_apps", _has_command("wmctrl"), "Via wmctrl"),
        ]


class LinuxCalendarAdapter(CalendarAdapter):
    """Linux calendar — stubs for now, Google Calendar is the primary option."""

    async def list_calendars(self) -> List[Dict[str, str]]:
        raise NotImplementedError(
            "Local calendar access is not yet supported on Linux. "
            "Please use Google Calendar integration instead."
        )

    async def get_events(self, *, start: str, end: str,
                         calendar_name: Optional[str] = None) -> List[CalendarEvent]:
        raise NotImplementedError(
            "Local calendar access is not yet supported on Linux. "
            "Please use Google Calendar integration instead."
        )

    async def create_event(self, *, calendar_name: str, title: str,
                           start: str, end: str,
                           description: str = "",
                           location: str = "") -> CalendarEvent:
        raise NotImplementedError(
            "Local calendar access is not yet supported on Linux. "
            "Please use Google Calendar integration instead."
        )

    async def delete_event(self, event_id: str, *,
                           calendar_name: Optional[str] = None) -> bool:
        raise NotImplementedError(
            "Local calendar access is not yet supported on Linux. "
            "Please use Google Calendar integration instead."
        )

    def capabilities(self) -> List[PlatformCapability]:
        return [
            PlatformCapability("local_calendar", False,
                               "Not yet supported on Linux — use Google Calendar"),
            PlatformCapability("create_event", False),
            PlatformCapability("delete_event", False),
            PlatformCapability("list_calendars", False),
        ]


class LinuxInfraAdapter(InfraAdapter):
    """Linux infrastructure — systemctl first, Docker fallback."""

    def _try_systemctl(self, service: str) -> bool:
        if not _has_command("systemctl"):
            return False
        try:
            result = subprocess.run(
                ["systemctl", "start", service],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _try_docker(self, name: str, image: str, port: int,
                    env: Optional[dict] = None,
                    extra_args: Optional[list] = None) -> bool:
        docker = shutil.which("docker")
        if not docker:
            return False
        try:
            result = subprocess.run([docker, "info"], capture_output=True, timeout=5)
            if result.returncode != 0:
                return False
        except Exception:
            return False

        check = subprocess.run(
            [docker, "ps", "-a", "--filter", f"name={name}", "--format", "{{.Status}}"],
            capture_output=True, text=True, timeout=5,
        )
        if check.stdout.strip():
            subprocess.run([docker, "start", name], capture_output=True, timeout=10)
            if _wait_for_port(port, max_wait=10):
                return True
            subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=10)

        cmd = [docker, "run", "-d", "--name", name, "-p", f"{port}:{port}", "--restart", "unless-stopped"]
        if env:
            for k, v in env.items():
                cmd.extend(["-e", f"{k}={v}"])
        cmd.append(image)
        if extra_args:
            cmd.extend(extra_args)
        try:
            subprocess.run(cmd, capture_output=True, timeout=30, check=True)
            return _wait_for_port(port, max_wait=15)
        except Exception:
            return False

    def start_service(self, service: str) -> bool:
        svc = service.lower()
        if svc == "redis":
            if _is_port_open(6379):
                return True
            # Try systemctl first
            if self._try_systemctl("redis-server") or self._try_systemctl("redis"):
                if _wait_for_port(6379, max_wait=10):
                    return True
            return self._try_docker("aa-redis", "redis:7-alpine", 6379,
                                    extra_args=["redis-server", "--appendonly", "yes"])
        elif svc in ("postgres", "postgresql"):
            if _is_port_open(5432):
                return True
            if self._try_systemctl("postgresql"):
                if _wait_for_port(5432, max_wait=10):
                    return True
            return self._try_docker(
                "aa-postgres", "postgres:16-alpine", 5432,
                env={"POSTGRES_USER": "agentadmin", "POSTGRES_PASSWORD": "localdev_secure_2026", "POSTGRES_DB": "code_horde"},
            )
        logger.warning("Unknown service: %s", service)
        return False

    def is_service_running(self, service: str) -> bool:
        ports = {"redis": 6379, "postgres": 5432, "postgresql": 5432}
        port = ports.get(service.lower())
        return _is_port_open(port) if port else False

    def install_service(self, service: str) -> bool:
        # Try apt-get (Debian/Ubuntu)
        if _has_command("apt-get"):
            mapping = {"redis": "redis-server", "postgres": "postgresql", "postgresql": "postgresql"}
            pkg = mapping.get(service.lower())
            if pkg:
                try:
                    subprocess.run(["sudo", "apt-get", "install", "-y", pkg],
                                   capture_output=True, timeout=120, check=True)
                    return True
                except Exception:
                    pass
        # Try dnf (Fedora/RHEL)
        if _has_command("dnf"):
            mapping = {"redis": "redis", "postgres": "postgresql-server", "postgresql": "postgresql-server"}
            pkg = mapping.get(service.lower())
            if pkg:
                try:
                    subprocess.run(["sudo", "dnf", "install", "-y", pkg],
                                   capture_output=True, timeout=120, check=True)
                    return True
                except Exception:
                    pass
        return False


class LinuxPathsAdapter(PathsAdapter):
    """Linux XDG-standard filesystem paths and pywebview backend."""

    @property
    def log_dir(self) -> Path:
        cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        return Path(cache) / "codehorde" / "logs"

    @property
    def config_dir(self) -> Path:
        config = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        return Path(config) / "codehorde"

    @property
    def temp_dir(self) -> Path:
        return Path(os.environ.get("TMPDIR", "/tmp")) / "codehorde"

    @property
    def webview_gui(self) -> str:
        return "gtk"


class LinuxSubprocessAdapter(SubprocessAdapter):
    """Linux subprocess / permission helpers."""

    def create_process_kwargs(self, *, isolated: bool = False) -> Dict[str, Any]:
        if isolated:
            return {"start_new_session": True}
        return {}

    def chmod_safe(self, path: Path, mode: int) -> None:
        path.chmod(mode)

    @property
    def shell_executable(self) -> str:
        return "/bin/bash"
