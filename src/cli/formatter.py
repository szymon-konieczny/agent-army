"""Rich terminal formatting for Code Horde CLI output."""

import json
from datetime import datetime
from typing import Any, Optional

# Try rich, fall back to plain text if not installed
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.columns import Columns
    from rich import box

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class CLIFormatter:
    """Handles all terminal output formatting.

    Uses `rich` if available, falls back to plain ANSI otherwise.
    """

    # ANSI escape codes (fallback)
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"

    # Agent role → colour mapping
    AGENT_COLORS: dict[str, str] = {
        "sentinel": "red",
        "builder": "green",
        "inspector": "yellow",
        "watcher": "cyan",
        "scout": "magenta",
        "scribe": "blue",
        "devops": "white",
        "commander": "bright_yellow",
    }

    def __init__(self) -> None:
        if HAS_RICH:
            self.console = Console()
        else:
            self.console = None  # type: ignore[assignment]

    # ── banners ──────────────────────────────────────────────────────

    def print_banner(self) -> None:
        """Print the welcome banner."""
        banner = r"""
    _                    _      _
   / \   __ _  ___ _ __ | |_   / \   _ __ _ __ ___  _   _
  / _ \ / _` |/ _ \ '_ \| __| / _ \ | '__| '_ ` _ \| | | |
 / ___ \ (_| |  __/ | | | |_ / ___ \| |  | | | | | | |_| |
/_/   \_\__, |\___|_| |_|\__/_/   \_\_|  |_| |_| |_|\__, |
        |___/                                        |___/
"""
        if self.console:
            self.console.print(
                Panel(
                    Text(banner, style="bold cyan"),
                    subtitle="[dim]Type /help for commands  •  Ctrl+C to exit[/dim]",
                    border_style="cyan",
                    expand=False,
                )
            )
        else:
            print(f"{self.CYAN}{self.BOLD}{banner}{self.RESET}")
            print(f"  {self.DIM}Type /help for commands  •  Ctrl+C to exit{self.RESET}")
            print()

    # ── health ───────────────────────────────────────────────────────

    def print_health(self, data: dict[str, Any]) -> None:
        """Format and print health check result."""
        status = data.get("status", "unknown")
        version = data.get("version", "?")
        components = data.get("components", {})

        if self.console:
            color = "green" if status == "healthy" else "red"
            lines = [
                f"[bold {color}]Status:[/bold {color}] {status}",
                f"[dim]Version:[/dim] {version}",
            ]
            for name, value in components.items():
                if isinstance(value, int):
                    lines.append(f"  {name}: {value}")
                else:
                    c = "green" if value == "healthy" else "red"
                    lines.append(f"  {name}: [{c}]{value}[/{c}]")
            self.console.print(Panel("\n".join(lines), title="Health", border_style=color))
        else:
            sym = "OK" if status == "healthy" else "!!"
            print(f"\n  [{sym}] {status}  (v{version})")
            for name, value in components.items():
                print(f"      {name}: {value}")
            print()

    # ── agents table ─────────────────────────────────────────────────

    def print_agents(self, data: dict[str, Any]) -> None:
        """Format and print the agents list."""
        agents = data.get("agents", {})
        total = data.get("total_agents", len(agents))

        if self.console:
            table = Table(
                title=f"Agents ({total})",
                box=box.ROUNDED,
                border_style="cyan",
                show_lines=True,
            )
            table.add_column("ID", style="bold")
            table.add_column("Name")
            table.add_column("Role")
            table.add_column("State", justify="center")
            table.add_column("Tasks OK", justify="right")
            table.add_column("Tasks Fail", justify="right")
            table.add_column("Uptime", justify="right")

            for agent_id, info in agents.items():
                state = info.get("state", "?")
                state_color = {
                    "idle": "green",
                    "busy": "yellow",
                    "paused": "dim",
                    "error": "red",
                    "offline": "red dim",
                }.get(state, "white")

                role = info.get("role", "")
                role_color = self.AGENT_COLORS.get(role, "white")

                uptime = info.get("uptime_seconds", 0)
                uptime_str = self._format_uptime(uptime)

                table.add_row(
                    agent_id,
                    info.get("name", ""),
                    f"[{role_color}]{role}[/{role_color}]",
                    f"[{state_color}]{state}[/{state_color}]",
                    str(info.get("tasks_completed", 0)),
                    str(info.get("tasks_failed", 0)),
                    uptime_str,
                )

            self.console.print(table)
        else:
            print(f"\n  Agents ({total}):")
            print(f"  {'ID':<20} {'Role':<12} {'State':<8} {'OK':<5} {'Fail':<5}")
            print(f"  {'─' * 60}")
            for agent_id, info in agents.items():
                print(
                    f"  {agent_id:<20} {info.get('role', ''):<12} "
                    f"{info.get('state', '?'):<8} "
                    f"{info.get('tasks_completed', 0):<5} "
                    f"{info.get('tasks_failed', 0):<5}"
                )
            print()

    # ── task result ──────────────────────────────────────────────────

    def print_task_submitted(self, data: dict[str, Any]) -> None:
        """Print task submission confirmation."""
        task_id = data.get("task_id", "?")
        agent = data.get("assigned_agent", "unassigned")
        status = data.get("status", "?")

        if self.console:
            self.console.print(
                Panel(
                    f"[bold]Task ID:[/bold] {task_id}\n"
                    f"[bold]Assigned:[/bold] {agent}\n"
                    f"[bold]Status:[/bold]  {status}",
                    title="Task Submitted",
                    border_style="green",
                )
            )
        else:
            print(f"\n  Task submitted: {task_id}")
            print(f"  Assigned to:    {agent}")
            print(f"  Status:         {status}\n")

    def print_task_status(self, data: dict[str, Any]) -> None:
        """Print task status details."""
        status = data.get("status", "?")
        color = {
            "pending": "yellow",
            "assigned": "cyan",
            "in_progress": "blue",
            "completed": "green",
            "failed": "red",
            "cancelled": "dim",
        }.get(status, "white")

        if self.console:
            lines = [
                f"[bold]Task:[/bold]    {data.get('task_id', '?')}",
                f"[bold]Status:[/bold]  [{color}]{status}[/{color}]",
                f"[bold]Agent:[/bold]   {data.get('assigned_agent', 'unassigned')}",
                f"[bold]Priority:[/bold] {data.get('priority', '?')}",
            ]
            desc = data.get("description", "")
            if desc:
                lines.append(f"[bold]Desc:[/bold]   {desc[:120]}")
            result = data.get("result")
            if result:
                lines.append(f"[bold]Result:[/bold]\n{json.dumps(result, indent=2)[:500]}")
            error = data.get("error")
            if error:
                lines.append(f"[bold red]Error:[/bold red] {error}")
            self.console.print(Panel("\n".join(lines), title="Task Detail", border_style=color))
        else:
            print(f"\n  Task: {data.get('task_id', '?')}")
            print(f"  Status:   {status}")
            print(f"  Agent:    {data.get('assigned_agent', 'unassigned')}")
            result = data.get("result")
            if result:
                print(f"  Result:   {json.dumps(result)[:200]}")
            error = data.get("error")
            if error:
                print(f"  Error:    {error}")
            print()

    # ── chat response ────────────────────────────────────────────────

    def print_chat_response(self, data: dict[str, Any]) -> None:
        """Print a chat/task response from an agent."""
        # If it came from /chat endpoint
        response = data.get("response") or data.get("message")
        agent = data.get("agent_id") or data.get("assigned_agent") or "system"

        if response:
            role_color = self.AGENT_COLORS.get(agent.split("-")[0], "white")
            if self.console:
                self.console.print(
                    f"[bold {role_color}]{agent}[/bold {role_color}] > {response}"
                )
            else:
                print(f"  {agent} > {response}")
        else:
            # Fallback: the response is a task submission
            self.print_task_submitted(data)

    # ── generic JSON ─────────────────────────────────────────────────

    def print_json(self, data: Any, title: str = "Response") -> None:
        """Pretty-print any JSON response."""
        if self.console:
            formatted = json.dumps(data, indent=2, default=str)
            self.console.print(
                Panel(Syntax(formatted, "json", theme="monokai"), title=title, border_style="dim")
            )
        else:
            print(json.dumps(data, indent=2, default=str))

    # ── messages ─────────────────────────────────────────────────────

    def info(self, message: str) -> None:
        if self.console:
            self.console.print(f"[cyan]>[/cyan] {message}")
        else:
            print(f"  > {message}")

    def success(self, message: str) -> None:
        if self.console:
            self.console.print(f"[green]✓[/green] {message}")
        else:
            print(f"  [OK] {message}")

    def warning(self, message: str) -> None:
        if self.console:
            self.console.print(f"[yellow]![/yellow] {message}")
        else:
            print(f"  [!] {message}")

    def error(self, message: str) -> None:
        if self.console:
            self.console.print(f"[bold red]✗[/bold red] {message}")
        else:
            print(f"  [ERR] {message}")

    # ── help ─────────────────────────────────────────────────────────

    def print_help(self) -> None:
        """Print available CLI commands."""
        commands = [
            ("/status", "System health check"),
            ("/agents", "List all agents and their states"),
            ("/task <desc>", "Submit a task (default priority 3)"),
            ("/task! <desc>", "Submit a HIGH priority task (priority 1)"),
            ("/poll <task_id>", "Check task status"),
            ("/wait <task_id>", "Wait for task to complete"),
            ("/chat <msg>", "Send a chat message to the orchestrator"),
            ("/to <agent> <msg>", "Send a message to a specific agent"),
            ("/json <endpoint>", "Raw GET request, print JSON"),
            ("/help", "Show this help"),
            ("/quit", "Exit the CLI"),
        ]

        if self.console:
            table = Table(box=box.SIMPLE, border_style="dim", show_header=False)
            table.add_column("Command", style="bold cyan", min_width=22)
            table.add_column("Description")
            for cmd, desc in commands:
                table.add_row(cmd, desc)
            self.console.print(Panel(table, title="Commands", border_style="cyan"))
        else:
            print("\n  Commands:")
            for cmd, desc in commands:
                print(f"    {cmd:<22} {desc}")
            print()

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Human-readable uptime."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h {m}m"
