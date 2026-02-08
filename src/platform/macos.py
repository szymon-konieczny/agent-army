"""macOS adapter implementations.

Wraps existing AppleScript / osascript / Shortcuts / Homebrew code that was
previously inlined in agents and desktop/app.py.  Nothing new is invented
here — the macOS behaviour is identical to before the refactor.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
import tempfile
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

async def _run_osascript(script: str, args: Optional[List[str]] = None) -> str:
    """Write *script* to a temp file and execute via ``osascript``."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".scpt", delete=False) as f:
        f.write(script)
        tmp = f.name
    try:
        cmd = ["osascript", tmp]
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
            raise RuntimeError(f"osascript failed (rc={proc.returncode}): {err}")
        return stdout.decode().strip()
    finally:
        os.unlink(tmp)


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


# ── macOS AppleScript templates (extracted from automator.py) ─────────

APPLESCRIPT_TEMPLATES = {
    "get_calendar_events": '''
tell application "Calendar"
    set today to current date
    set todayStart to today - (time of today)
    set todayEnd to todayStart + (1 * days)
    set output to ""
    repeat with cal in calendars
        set evts to (every event of cal whose start date >= todayStart and start date < todayEnd)
        repeat with evt in evts
            set output to output & (summary of evt) & " | " & (start date of evt) & " - " & (end date of evt) & linefeed
        end repeat
    end repeat
    return output
end tell
''',
    "get_frontmost_app": '''
tell application "System Events"
    set frontApp to name of first application process whose frontmost is true
    try
        set frontWindow to name of front window of (first application process whose frontmost is true)
    on error
        set frontWindow to "No window"
    end try
    return frontApp & " — " & frontWindow
end tell
''',
    "list_open_apps": '''
tell application "System Events"
    set appList to ""
    repeat with proc in (every application process whose visible is true)
        set appList to appList & (name of proc) & linefeed
    end repeat
    return appList
end tell
''',
    "notification": '''
on run argv
    set notifTitle to item 1 of argv
    set notifBody to item 2 of argv
    display notification notifBody with title notifTitle
end run
''',
    "clipboard_get": 'return (the clipboard as text)',
    "clipboard_set": '''
on run argv
    set the clipboard to (item 1 of argv)
    return "Clipboard set"
end run
''',
}

# ── macOS Calendar AppleScript templates (extracted from scheduler.py) ─

MACOS_CALENDAR_SCRIPTS = {
    "get_today_events": '''
tell application "Calendar"
    set today to current date
    set todayStart to today - (time of today)
    set todayEnd to todayStart + (1 * days)
    set output to ""
    repeat with cal in calendars
        set evts to (every event of cal whose start date >= todayStart and start date < todayEnd)
        repeat with evt in evts
            set output to output & "- " & (summary of evt) & " | " & (start date of evt) & " → " & (end date of evt) & " | Calendar: " & (name of cal) & linefeed
        end repeat
    end repeat
    if output is "" then
        return "No events scheduled for today."
    end if
    return output
end tell
''',
    "get_tomorrow_events": '''
tell application "Calendar"
    set today to current date
    set tomorrowStart to (today - (time of today)) + (1 * days)
    set tomorrowEnd to tomorrowStart + (1 * days)
    set output to ""
    repeat with cal in calendars
        set evts to (every event of cal whose start date >= tomorrowStart and start date < tomorrowEnd)
        repeat with evt in evts
            set output to output & "- " & (summary of evt) & " | " & (start date of evt) & " → " & (end date of evt) & " | Calendar: " & (name of cal) & linefeed
        end repeat
    end repeat
    if output is "" then
        return "No events scheduled for tomorrow."
    end if
    return output
end tell
''',
    "get_week_events": '''
tell application "Calendar"
    set today to current date
    set weekStart to today - (time of today)
    set weekEnd to weekStart + (7 * days)
    set output to ""
    repeat with cal in calendars
        set evts to (every event of cal whose start date >= weekStart and start date < weekEnd)
        repeat with evt in evts
            set output to output & "- " & (summary of evt) & " | " & (start date of evt) & " → " & (end date of evt) & " | " & (name of cal) & linefeed
        end repeat
    end repeat
    if output is "" then
        return "No events this week."
    end if
    return output
end tell
''',
    "create_event": '''
on run argv
    set calName to item 1 of argv
    set evtTitle to item 2 of argv
    set evtStart to date (item 3 of argv)
    set evtEnd to date (item 4 of argv)
    set evtNotes to ""
    if (count of argv) > 4 then
        set evtNotes to item 5 of argv
    end if
    tell application "Calendar"
        tell calendar calName
            make new event with properties {summary:evtTitle, start date:evtStart, end date:evtEnd, description:evtNotes}
        end tell
    end tell
    return "Created: " & evtTitle & " on " & evtStart
end run
''',
    "list_calendars": '''
tell application "Calendar"
    set output to ""
    repeat with cal in calendars
        set output to output & "- " & (name of cal) & " (" & (description of cal) & ")" & linefeed
    end repeat
    return output
end tell
''',
    "delete_event": '''
on run argv
    set targetTitle to item 1 of argv
    tell application "Calendar"
        set today to current date
        set todayStart to today - (time of today)
        set todayEnd to todayStart + (1 * days)
        repeat with cal in calendars
            set evts to (every event of cal whose summary is targetTitle and start date >= todayStart and start date < todayEnd)
            repeat with evt in evts
                delete evt
            end repeat
        end repeat
    end tell
    return "Deleted event: " & targetTitle
end run
''',
}


# ═══════════════════════════════════════════════════════════════════════
# Adapter implementations
# ═══════════════════════════════════════════════════════════════════════

class MacDesktopAdapter(DesktopAdapter):
    """macOS desktop automation via AppleScript / osascript / Shortcuts CLI."""

    async def execute_script(self, script: str, *, script_type: str = "auto",
                             args: Optional[List[str]] = None) -> str:
        if script_type in ("auto", "applescript"):
            return await _run_osascript(script, args)
        elif script_type == "bash":
            proc = await asyncio.create_subprocess_shell(
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip()
        raise ValueError(f"Unsupported script type on macOS: {script_type}")

    async def clipboard_get(self) -> str:
        return await _run_osascript(APPLESCRIPT_TEMPLATES["clipboard_get"])

    async def clipboard_set(self, text: str) -> None:
        await _run_osascript(APPLESCRIPT_TEMPLATES["clipboard_set"], [text])

    async def show_notification(self, title: str, body: str) -> None:
        await _run_osascript(APPLESCRIPT_TEMPLATES["notification"], [title, body])

    async def list_shortcuts(self) -> List[Dict[str, str]]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "shortcuts", "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            lines = stdout.decode().strip().splitlines()
            return [{"name": line.strip(), "id": line.strip()} for line in lines if line.strip()]
        except FileNotFoundError:
            return []

    async def run_shortcut(self, name: str, *,
                           input_path: Optional[str] = None,
                           output_path: Optional[str] = None) -> str:
        cmd = ["shortcuts", "run", name]
        if input_path:
            cmd.extend(["--input-path", input_path])
        if output_path:
            cmd.extend(["--output-path", output_path])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Shortcut '{name}' failed: {stderr.decode().strip()}")
        return stdout.decode().strip()

    async def get_frontmost_app(self) -> str:
        return await _run_osascript(APPLESCRIPT_TEMPLATES["get_frontmost_app"])

    async def list_open_apps(self) -> List[str]:
        raw = await _run_osascript(APPLESCRIPT_TEMPLATES["list_open_apps"])
        return [a.strip() for a in raw.splitlines() if a.strip()]

    def capabilities(self) -> List[PlatformCapability]:
        has_shortcuts = shutil.which("shortcuts") is not None
        return [
            PlatformCapability("applescript", True),
            PlatformCapability("osascript", True),
            PlatformCapability("shortcuts", has_shortcuts,
                               "" if has_shortcuts else "Shortcuts CLI not found"),
            PlatformCapability("clipboard", True),
            PlatformCapability("notifications", True),
            PlatformCapability("frontmost_app", True),
            PlatformCapability("list_open_apps", True),
        ]


class MacCalendarAdapter(CalendarAdapter):
    """macOS Calendar.app access via AppleScript."""

    async def list_calendars(self) -> List[Dict[str, str]]:
        raw = await _run_osascript(MACOS_CALENDAR_SCRIPTS["list_calendars"])
        results = []
        for line in raw.splitlines():
            line = line.strip().lstrip("- ")
            if line:
                results.append({"name": line, "description": ""})
        return results

    async def get_events(self, *, start: str, end: str,
                         calendar_name: Optional[str] = None) -> List[CalendarEvent]:
        # Use today/tomorrow/week helper or the range script from automator
        raw = await _run_osascript(MACOS_CALENDAR_SCRIPTS["get_today_events"])
        events = []
        for line in raw.splitlines():
            line = line.strip().lstrip("- ")
            if not line or line.startswith("No events"):
                continue
            parts = [p.strip() for p in line.split("|")]
            title = parts[0] if len(parts) > 0 else "Unknown"
            time_range = parts[1] if len(parts) > 1 else ""
            cal = parts[2].replace("Calendar:", "").strip() if len(parts) > 2 else ""
            times = [t.strip() for t in time_range.split("→")] if "→" in time_range else [time_range, ""]
            events.append(CalendarEvent(
                id=title,
                title=title,
                start=times[0],
                end=times[1] if len(times) > 1 else "",
                calendar_name=cal,
            ))
        return events

    async def create_event(self, *, calendar_name: str, title: str,
                           start: str, end: str,
                           description: str = "",
                           location: str = "") -> CalendarEvent:
        args = [calendar_name, title, start, end]
        if description:
            args.append(description)
        result = await _run_osascript(MACOS_CALENDAR_SCRIPTS["create_event"], args)
        logger.info("Calendar event created: %s", result)
        return CalendarEvent(
            id=title, title=title, start=start, end=end,
            calendar_name=calendar_name, description=description, location=location,
        )

    async def delete_event(self, event_id: str, *,
                           calendar_name: Optional[str] = None) -> bool:
        try:
            await _run_osascript(MACOS_CALENDAR_SCRIPTS["delete_event"], [event_id])
            return True
        except RuntimeError:
            return False

    def capabilities(self) -> List[PlatformCapability]:
        return [
            PlatformCapability("local_calendar", True, "macOS Calendar.app via AppleScript"),
            PlatformCapability("create_event", True),
            PlatformCapability("delete_event", True),
            PlatformCapability("list_calendars", True),
        ]


class MacInfraAdapter(InfraAdapter):
    """macOS infrastructure — Homebrew-first, Docker fallback."""

    def _has_brew(self) -> bool:
        return shutil.which("brew") is not None

    def _brew_service_installed(self, service: str) -> bool:
        try:
            result = subprocess.run(
                ["brew", "list", service],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _try_docker_single(self, name: str, image: str, port: int,
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
            return self._start_redis()
        elif svc in ("postgres", "postgresql"):
            return self._start_postgres()
        logger.warning("Unknown service: %s", service)
        return False

    def is_service_running(self, service: str) -> bool:
        ports = {"redis": 6379, "postgres": 5432, "postgresql": 5432}
        port = ports.get(service.lower())
        return _is_port_open(port) if port else False

    def install_service(self, service: str) -> bool:
        if not self._has_brew():
            return False
        mapping = {"redis": "redis", "postgres": "postgresql@16", "postgresql": "postgresql@16"}
        formula = mapping.get(service.lower())
        if not formula:
            return False
        try:
            subprocess.run(["brew", "install", formula], capture_output=True, timeout=300, check=True)
            return True
        except Exception:
            return False

    # -- private helpers extracted from desktop/app.py --

    def _start_redis(self) -> bool:
        if _is_port_open(6379):
            return True
        if self._has_brew():
            if self._brew_service_installed("redis"):
                subprocess.run(["brew", "services", "start", "redis"], capture_output=True, timeout=10)
                if _wait_for_port(6379, max_wait=10):
                    return True
            else:
                subprocess.run(["brew", "install", "redis"], capture_output=True, timeout=120)
                subprocess.run(["brew", "services", "start", "redis"], capture_output=True, timeout=10)
                if _wait_for_port(6379, max_wait=10):
                    return True
        return self._try_docker_single("aa-redis", "redis:7-alpine", 6379, extra_args=["redis-server", "--appendonly", "yes"])

    def _start_postgres(self) -> bool:
        if _is_port_open(5432):
            return True
        if self._has_brew():
            for pg in ["postgresql@16", "postgresql@15", "postgresql@14", "postgresql"]:
                if self._brew_service_installed(pg):
                    subprocess.run(["brew", "services", "start", pg], capture_output=True, timeout=10)
                    if _wait_for_port(5432, max_wait=10):
                        return True
            subprocess.run(["brew", "install", "postgresql@16"], capture_output=True, timeout=300)
            subprocess.run(["brew", "services", "start", "postgresql@16"], capture_output=True, timeout=10)
            if _wait_for_port(5432, max_wait=15):
                return True
        return self._try_docker_single(
            "aa-postgres", "postgres:16-alpine", 5432,
            env={"POSTGRES_USER": "agentadmin", "POSTGRES_PASSWORD": "localdev_secure_2026", "POSTGRES_DB": "agent_army"},
        )


class MacPathsAdapter(PathsAdapter):
    """macOS filesystem paths and pywebview backend."""

    @property
    def log_dir(self) -> Path:
        return Path.home() / "Library" / "Logs" / "AgentArmy"

    @property
    def config_dir(self) -> Path:
        return Path.home() / "Library" / "Application Support" / "AgentArmy"

    @property
    def temp_dir(self) -> Path:
        return Path(os.environ.get("TMPDIR", "/tmp")) / "agentarmy"

    @property
    def webview_gui(self) -> str:
        return "cocoa"


class MacSubprocessAdapter(SubprocessAdapter):
    """macOS subprocess / permission helpers."""

    def create_process_kwargs(self, *, isolated: bool = False) -> Dict[str, Any]:
        if isolated:
            return {"start_new_session": True}
        return {}

    def chmod_safe(self, path: Path, mode: int) -> None:
        path.chmod(mode)

    @property
    def shell_executable(self) -> str:
        return "/bin/bash"
