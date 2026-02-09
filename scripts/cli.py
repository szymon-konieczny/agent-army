#!/usr/bin/env python3
"""Entry point for the Code Horde interactive terminal.

Usage:
    python scripts/cli.py                     # connect to localhost:8000
    python scripts/cli.py --url http://host   # connect to a different host
    python scripts/cli.py --exec "/status"    # run a single command and exit
"""

import argparse
import asyncio
import os
import sys

# Ensure project root is on path so `from src.…` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli.client import AgentArmyCLI
from src.cli.formatter import CLIFormatter
from src.cli.repl import InteractiveREPL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Code Horde — Interactive Terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                          # interactive mode\n"
            "  %(prog)s --exec '/status'         # one-shot command\n"
            "  %(prog)s --exec '/task scan repo' # submit a task and exit\n"
            "  %(prog)s --url http://10.0.0.5:8000\n"
        ),
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("AGENTARMY_API_URL", "http://localhost:8000"),
        help="Code Horde API base URL (default: $AGENTARMY_API_URL or localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--exec",
        dest="execute",
        metavar="CMD",
        help="Execute a single command and exit (e.g. '/status', '/agents')",
    )
    return parser.parse_args()


async def one_shot(url: str, timeout: float, command: str) -> None:
    """Execute a single command, print output, and exit."""
    repl = InteractiveREPL(base_url=url, timeout=timeout)
    await repl.api.connect()
    try:
        await repl._dispatch(command)
    finally:
        await repl.api.close()


async def interactive(url: str, timeout: float) -> None:
    """Run the full interactive REPL."""
    repl = InteractiveREPL(base_url=url, timeout=timeout)
    await repl.run()


def main() -> None:
    args = parse_args()

    if args.execute:
        asyncio.run(one_shot(args.url, args.timeout, args.execute))
    else:
        asyncio.run(interactive(args.url, args.timeout))


if __name__ == "__main__":
    main()
