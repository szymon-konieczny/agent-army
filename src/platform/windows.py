"""Windows adapter implementations.

Provides PowerShell-based desktop automation, Docker-only infrastructure,
and Windows-standard paths.  Features not yet implemented return clear
``NotImplementedError`` messages so agents can fall back gracefully.
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


async def _run_powershell(script: str, args: Optional[List[str]] = None) -> str:
    """Execute a PowerShell script and return stdout."""
    cmd = [
        "powershell.exe", "-NoProfile", "-NonInteractive",
        "-ExecutionPolicy", "Bypass", "-Command", script,
    ]
    if args:
        cmd.extend(args)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode().strip()
        raise RuntimeError(f"PowerShell failed (rc={proc.returncode}): {err}")
    return stdout.decode().strip()


# ═══════════════════════════════════════════════════════════════════════
# Adapter implementations
# ═══════════════════════════════════════════════════════════════════════

class WindowsDesktopAdapter(DesktopAdapter):
    """Windows desktop automation via PowerShell."""

    async def execute_script(self, script: str, *, script_type: str = "auto",
                             args: Optional[List[str]] = None) -> str:
        if script_type in ("auto", "powershell"):
            return await _run_powershell(script, args)
        elif script_type == "applescript":
            raise NotImplementedError(
                "AppleScript is not available on Windows. "
                "Use PowerShell or bash script_type instead."
            )
        elif script_type == "bash":
            # Git Bash / WSL may be available
            proc = await asyncio.create_subprocess_shell(
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip()
        raise ValueError(f"Unsupported script type on Windows: {script_type}")

    async def clipboard_get(self) -> str:
        return await _run_powershell("Get-Clipboard")

    async def clipboard_set(self, text: str) -> None:
        # Escape single quotes for PowerShell
        escaped = text.replace("'", "''")
        await _run_powershell(f"Set-Clipboard -Value '{escaped}'")

    async def show_notification(self, title: str, body: str) -> None:
        # Windows 10+ toast notification via PowerShell
        ps_script = f'''
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$textNodes = $template.GetElementsByTagName("text")
$textNodes.Item(0).AppendChild($template.CreateTextNode("{title}")) > $null
$textNodes.Item(1).AppendChild($template.CreateTextNode("{body}")) > $null
$toast = [Windows.UI.Notifications.ToastNotification]::new($template)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("AgentArmy").Show($toast)
'''
        try:
            await _run_powershell(ps_script)
        except RuntimeError:
            # Fallback to BurntToast module or simple message box
            logger.debug("Toast notification failed, trying fallback")
            await _run_powershell(
                f'Add-Type -AssemblyName System.Windows.Forms; '
                f'[System.Windows.Forms.MessageBox]::Show("{body}", "{title}")'
            )

    async def list_shortcuts(self) -> List[Dict[str, str]]:
        # No macOS Shortcuts equivalent yet — could list Task Scheduler tasks later
        return []

    async def run_shortcut(self, name: str, *,
                           input_path: Optional[str] = None,
                           output_path: Optional[str] = None) -> str:
        raise NotImplementedError(
            "macOS Shortcuts are not available on Windows. "
            "Consider using Task Scheduler or PowerShell scripts instead."
        )

    async def get_frontmost_app(self) -> str:
        return await _run_powershell(
            "(Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | "
            "Sort-Object -Property CPU -Descending | Select-Object -First 1).MainWindowTitle"
        )

    async def list_open_apps(self) -> List[str]:
        raw = await _run_powershell(
            "Get-Process | Where-Object {$_.MainWindowTitle -ne ''} | "
            "Select-Object -ExpandProperty MainWindowTitle"
        )
        return [line.strip() for line in raw.splitlines() if line.strip()]

    def capabilities(self) -> List[PlatformCapability]:
        return [
            PlatformCapability("applescript", False, "Not available on Windows"),
            PlatformCapability("osascript", False, "Not available on Windows"),
            PlatformCapability("powershell", True),
            PlatformCapability("shortcuts", False, "macOS Shortcuts not available on Windows"),
            PlatformCapability("clipboard", True, "Via PowerShell Get-Clipboard / Set-Clipboard"),
            PlatformCapability("notifications", True, "Via PowerShell toast notifications"),
            PlatformCapability("frontmost_app", True, "Via Get-Process"),
            PlatformCapability("list_open_apps", True, "Via Get-Process"),
        ]


class WindowsCalendarAdapter(CalendarAdapter):
    """Windows calendar — stubs for now, Google Calendar is the primary option."""

    async def list_calendars(self) -> List[Dict[str, str]]:
        raise NotImplementedError(
            "Local calendar access is not yet supported on Windows. "
            "Please use Google Calendar integration instead."
        )

    async def get_events(self, *, start: str, end: str,
                         calendar_name: Optional[str] = None) -> List[CalendarEvent]:
        raise NotImplementedError(
            "Local calendar access is not yet supported on Windows. "
            "Please use Google Calendar integration instead."
        )

    async def create_event(self, *, calendar_name: str, title: str,
                           start: str, end: str,
                           description: str = "",
                           location: str = "") -> CalendarEvent:
        raise NotImplementedError(
            "Local calendar access is not yet supported on Windows. "
            "Please use Google Calendar integration instead."
        )

    async def delete_event(self, event_id: str, *,
                           calendar_name: Optional[str] = None) -> bool:
        raise NotImplementedError(
            "Local calendar access is not yet supported on Windows. "
            "Please use Google Calendar integration instead."
        )

    def capabilities(self) -> List[PlatformCapability]:
        return [
            PlatformCapability("local_calendar", False,
                               "Not yet supported on Windows — use Google Calendar"),
            PlatformCapability("create_event", False),
            PlatformCapability("delete_event", False),
            PlatformCapability("list_calendars", False),
        ]


class WindowsInfraAdapter(InfraAdapter):
    """Windows infrastructure — Docker-only (no Homebrew)."""

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
            return self._try_docker("aa-redis", "redis:7-alpine", 6379,
                                    extra_args=["redis-server", "--appendonly", "yes"])
        elif svc in ("postgres", "postgresql"):
            if _is_port_open(5432):
                return True
            return self._try_docker(
                "aa-postgres", "postgres:16-alpine", 5432,
                env={"POSTGRES_USER": "agentadmin", "POSTGRES_PASSWORD": "localdev_secure_2026", "POSTGRES_DB": "agent_army"},
            )
        logger.warning("Unknown service: %s", service)
        return False

    def is_service_running(self, service: str) -> bool:
        ports = {"redis": 6379, "postgres": 5432, "postgresql": 5432}
        port = ports.get(service.lower())
        return _is_port_open(port) if port else False

    def install_service(self, service: str) -> bool:
        logger.info("Windows install_service: use Docker Desktop to run %s", service)
        return False  # User must install Docker Desktop manually


class WindowsPathsAdapter(PathsAdapter):
    """Windows filesystem paths and pywebview backend."""

    @property
    def log_dir(self) -> Path:
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(appdata) / "AgentArmy" / "logs"

    @property
    def config_dir(self) -> Path:
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(appdata) / "AgentArmy"

    @property
    def temp_dir(self) -> Path:
        return Path(os.environ.get("TEMP", os.environ.get("TMP", "C:\\Temp"))) / "agentarmy"

    @property
    def webview_gui(self) -> str:
        return "edgechromium"


class WindowsSubprocessAdapter(SubprocessAdapter):
    """Windows subprocess / permission helpers."""

    def create_process_kwargs(self, *, isolated: bool = False) -> Dict[str, Any]:
        if isolated:
            import subprocess as _sp
            return {"creationflags": _sp.CREATE_NEW_PROCESS_GROUP}
        return {}

    def chmod_safe(self, path: Path, mode: int) -> None:
        # Windows doesn't use POSIX permissions — no-op with debug log
        logger.debug("chmod_safe: skipped on Windows for %s (mode=%o)", path, mode)

    @property
    def shell_executable(self) -> str:
        return "cmd.exe"
