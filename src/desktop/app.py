"""
AgentArmy — cross-platform native desktop app.

Wraps the Command Center dashboard in a native window via pywebview.

When launched (double-click AgentArmy.app or `make app-open`), it automatically:
  1. Starts PostgreSQL + Redis (via platform-specific service manager)
  2. Boots the FastAPI server in a background thread
  3. Opens the dashboard in a native window (Cocoa / EdgeChromium / GTK)

No terminal, no Docker Desktop, no manual steps required.
"""

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Optional

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.platform import get_infra_adapter, get_paths_adapter


# ── Helpers ──────────────────────────────────────────────────────────

def is_api_running(host: str = "127.0.0.1", port: int = 8000, timeout: float = 1.0) -> bool:
    """Check if the AgentArmy API is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError, socket.timeout):
        return False


def _is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    """Check if a TCP port is open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError, socket.timeout):
        return False


def wait_for_api(host: str = "127.0.0.1", port: int = 8000, max_wait: float = 30.0) -> bool:
    """Block until the API is reachable (or timeout)."""
    start = time.monotonic()
    while time.monotonic() - start < max_wait:
        if is_api_running(host, port):
            return True
        time.sleep(0.3)
    return False


def _wait_for_port(port: int, max_wait: float = 15.0) -> bool:
    """Wait until a port becomes reachable."""
    start = time.monotonic()
    while time.monotonic() - start < max_wait:
        if _is_port_open(port):
            return True
        time.sleep(0.5)
    return False


def start_infrastructure() -> bool:
    """Start PostgreSQL using platform-specific service management.

    Redis is NOT required — the app uses an in-memory cache automatically.
    Only PostgreSQL is needed as an external service.

    Returns True if PostgreSQL is reachable.
    """
    print("Checking infrastructure...")
    infra = get_infra_adapter()

    pg_ok = infra.start_service("postgres")

    # Redis is optional — try to start but don't block.
    redis_ok = infra.is_service_running("redis")
    if not redis_ok:
        redis_ok = infra.start_service("redis")

    if pg_ok:
        extra = " + Redis" if redis_ok else " (in-memory cache)"
        print(f"Infrastructure ready: PostgreSQL{extra}.")
    else:
        print("WARNING: PostgreSQL not available.")

    return pg_ok


# ── Embedded server (standalone mode) ────────────────────────────────

def start_server_thread(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Start uvicorn in a daemon thread."""
    import uvicorn

    config = uvicorn.Config(
        "src.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True, name="uvicorn")
    thread.start()
    return thread


# ── System tray menu (macOS menu bar items) ──────────────────────────

class AppAPI:
    """Exposed to JavaScript in the webview via `window.pywebview.api`."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def get_api_url(self) -> str:
        """Return the API base URL so JS can discover it."""
        return self.base_url

    def ping(self) -> dict:
        """Simple health-check callable from JS."""
        return {"status": "ok", "source": "desktop"}


# ── Window creation ──────────────────────────────────────────────────

def create_window(
    url: str,
    title: str = "AgentArmy — Command Center",
    width: int = 1400,
    height: int = 900,
) -> None:
    """Create and show the native desktop window (Cocoa / EdgeChromium / GTK)."""
    import webview

    app_api = AppAPI(base_url=url)
    gui_backend = get_paths_adapter().webview_gui

    window = webview.create_window(
        title=title,
        url=url,
        width=width,
        height=height,
        min_size=(800, 600),
        resizable=True,
        background_color="#0f172a",  # slate-950 — matches dashboard
        js_api=app_api,
        text_select=True,
    )

    webview.start(
        debug=os.environ.get("AGENTARMY_DEBUG", "").lower() in ("1", "true"),
        gui=gui_backend,
    )


# ── Main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgentArmy — Desktop App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("AGENTARMY_API_URL", "http://localhost:8000"),
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        default=True,  # Always standalone by default
        help="Auto-start infrastructure + API server (default: True)",
    )
    parser.add_argument(
        "--client-only",
        action="store_true",
        help="Connect to an already-running API (skip infrastructure)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port (default: 8000)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1400,
        help="Window width (default: 1400)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="Window height (default: 900)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = args.url

    if args.client_only:
        # Client mode — just connect to an existing API
        if not is_api_running(port=args.port):
            print(
                "AgentArmy API is not running.\n"
                "  Start it with 'make dev' or remove --client-only\n"
            )
            sys.exit(1)
        url = f"http://localhost:{args.port}"
    else:
        # Standalone mode (default) — start everything automatically
        # 1. Start infrastructure (Homebrew-first, Docker fallback)
        if not start_infrastructure():
            print(
                "WARNING: Infrastructure not fully available.\n"
                "  The app will start but some features may not work.\n"
                "  Install with: brew install postgresql@16 redis\n"
            )

        # 2. Start the API server
        if is_api_running(port=args.port):
            print(f"API already running on port {args.port}")
        else:
            print(f"Starting API server on port {args.port}...")
            start_server_thread(port=args.port)
            if not wait_for_api(port=args.port):
                print("ERROR: API server failed to start within 30s")
                sys.exit(1)
            print("API server ready.")
        url = f"http://localhost:{args.port}"

    print(f"Opening Command Center: {url}")
    create_window(url=url, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
