"""Interactive REPL (Read-Eval-Print Loop) for the Code Horde terminal."""

import asyncio
import readline
import shlex
import signal
import sys
from typing import Optional

from src.cli.client import Code HordeCLI, APIError
from src.cli.formatter import CLIFormatter


# ── readline history ─────────────────────────────────────────────────
HISTORY_FILE = ".codehorde_history"


def _setup_readline() -> None:
    """Configure tab completion and persistent history."""
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(500)

    commands = [
        "/status",
        "/agents",
        "/task",
        "/task!",
        "/poll",
        "/wait",
        "/chat",
        "/to",
        "/json",
        "/help",
        "/quit",
    ]

    def completer(text: str, state: int) -> Optional[str]:
        matches = [c for c in commands if c.startswith(text)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")


class InteractiveREPL:
    """Full-featured interactive terminal for Code Horde.

    Supports slash commands and free-form chat:
      /status           → system health
      /agents           → agent list
      /task <text>      → submit task
      /task! <text>     → submit HIGH priority task
      /poll <id>        → check task status
      /wait <id>        → block until task done
      /chat <msg>       → orchestrator chat
      /to <agent> <msg> → direct message to specific agent
      /json <path>      → raw API GET
      /help             → command list
      /quit             → exit

    Anything without a leading / is treated as a /chat message.
    """

    PROMPT = "\033[36magent-army\033[0m > "

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ) -> None:
        self.api = Code HordeCLI(base_url=base_url, timeout=timeout)
        self.fmt = CLIFormatter()
        self._running = False

    # ── main loop ────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the interactive REPL."""
        _setup_readline()
        self._running = True

        # Connect
        await self.api.connect()

        # Print banner
        self.fmt.print_banner()

        # Check connectivity
        alive = await self.api.is_alive()
        if alive:
            self.fmt.success("Connected to Code Horde API")
        else:
            self.fmt.warning(
                f"Cannot reach Code Horde at {self.api.base_url} — "
                "start with 'make dev' first"
            )

        # Loop
        while self._running:
            try:
                line = input(self.PROMPT).strip()
                if not line:
                    continue
                await self._dispatch(line)
            except (KeyboardInterrupt, EOFError):
                print()
                self.fmt.info("Goodbye!")
                break
            except Exception as exc:
                self.fmt.error(str(exc))

        # Save history & close
        try:
            readline.write_history_file(HISTORY_FILE)
        except Exception:
            pass
        await self.api.close()

    # ── command dispatch ─────────────────────────────────────────────

    async def _dispatch(self, line: str) -> None:
        """Route a line of input to the correct handler."""
        if line.startswith("/"):
            parts = line.split(None, 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            handler = {
                "/status": self._cmd_status,
                "/health": self._cmd_status,
                "/agents": self._cmd_agents,
                "/task": self._cmd_task,
                "/task!": self._cmd_task_high,
                "/poll": self._cmd_poll,
                "/wait": self._cmd_wait,
                "/chat": self._cmd_chat,
                "/to": self._cmd_direct,
                "/json": self._cmd_json,
                "/help": self._cmd_help,
                "/quit": self._cmd_quit,
                "/exit": self._cmd_quit,
                "/q": self._cmd_quit,
            }.get(command)

            if handler:
                await handler(args)
            else:
                self.fmt.warning(f"Unknown command: {command}  (type /help)")
        else:
            # Free-form text → treat as chat
            await self._cmd_chat(line)

    # ── command handlers ─────────────────────────────────────────────

    async def _cmd_status(self, _args: str) -> None:
        """Handle /status."""
        try:
            data = await self.api.health()
            self.fmt.print_health(data)
        except ConnectionError as exc:
            self.fmt.error(f"API unreachable: {exc}")
        except APIError as exc:
            self.fmt.error(f"API error: {exc}")

    async def _cmd_agents(self, _args: str) -> None:
        """Handle /agents."""
        try:
            data = await self.api.list_agents()
            self.fmt.print_agents(data)
        except ConnectionError as exc:
            self.fmt.error(f"API unreachable: {exc}")
        except APIError as exc:
            self.fmt.error(f"API error: {exc}")

    async def _cmd_task(self, args: str) -> None:
        """Handle /task <description>."""
        if not args.strip():
            self.fmt.warning("Usage: /task <description>")
            return
        try:
            data = await self.api.submit_task(args.strip(), priority=3)
            self.fmt.print_task_submitted(data)
        except (ConnectionError, APIError) as exc:
            self.fmt.error(str(exc))

    async def _cmd_task_high(self, args: str) -> None:
        """Handle /task! <description> (high priority)."""
        if not args.strip():
            self.fmt.warning("Usage: /task! <description>")
            return
        try:
            data = await self.api.submit_task(args.strip(), priority=1, tags=["urgent"])
            self.fmt.print_task_submitted(data)
        except (ConnectionError, APIError) as exc:
            self.fmt.error(str(exc))

    async def _cmd_poll(self, args: str) -> None:
        """Handle /poll <task_id>."""
        task_id = args.strip()
        if not task_id:
            self.fmt.warning("Usage: /poll <task_id>")
            return
        try:
            data = await self.api.get_task(task_id)
            self.fmt.print_task_status(data)
        except APIError as exc:
            if exc.status_code == 404:
                self.fmt.warning(f"Task not found: {task_id}")
            else:
                self.fmt.error(str(exc))
        except ConnectionError as exc:
            self.fmt.error(str(exc))

    async def _cmd_wait(self, args: str) -> None:
        """Handle /wait <task_id> — block until task completes."""
        task_id = args.strip()
        if not task_id:
            self.fmt.warning("Usage: /wait <task_id>")
            return
        self.fmt.info(f"Waiting for task {task_id}...")
        try:
            data = await self.api.wait_for_task(task_id, timeout=300.0)
            self.fmt.print_task_status(data)
        except TimeoutError:
            self.fmt.warning(f"Timeout waiting for {task_id}")
        except (ConnectionError, APIError) as exc:
            self.fmt.error(str(exc))

    async def _cmd_chat(self, args: str) -> None:
        """Handle /chat <message> or free-form text."""
        message = args.strip()
        if not message:
            self.fmt.warning("Usage: /chat <message>  (or just type anything)")
            return
        try:
            data = await self.api.chat(message)
            self.fmt.print_chat_response(data)
        except (ConnectionError, APIError) as exc:
            self.fmt.error(str(exc))

    async def _cmd_direct(self, args: str) -> None:
        """Handle /to <agent_id> <message>."""
        parts = args.split(None, 1)
        if len(parts) < 2:
            self.fmt.warning("Usage: /to <agent_id> <message>")
            return
        agent_id, message = parts
        try:
            data = await self.api.chat(message, agent_id=agent_id)
            self.fmt.print_chat_response(data)
        except (ConnectionError, APIError) as exc:
            self.fmt.error(str(exc))

    async def _cmd_json(self, args: str) -> None:
        """Handle /json <path> — raw GET request."""
        path = args.strip()
        if not path:
            self.fmt.warning("Usage: /json <path>  (e.g. /json /health)")
            return
        if not path.startswith("/"):
            path = "/" + path
        try:
            data = await self.api._request("GET", path)
            self.fmt.print_json(data, title=f"GET {path}")
        except (ConnectionError, APIError) as exc:
            self.fmt.error(str(exc))

    async def _cmd_help(self, _args: str) -> None:
        """Handle /help."""
        self.fmt.print_help()

    async def _cmd_quit(self, _args: str) -> None:
        """Handle /quit."""
        self.fmt.info("Goodbye!")
        self._running = False
