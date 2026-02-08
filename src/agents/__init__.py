"""Agent implementations for AgentArmy system.

Provides specialized agents for various operational roles:
- Sentinel: Security scanning and vulnerability detection
- Builder: Code generation and CI/CD management
- Inspector: Quality assurance and testing
- Watcher: System monitoring and health checks
- Scout: Research and competitive analysis
- Scribe: Documentation generation
- DevOps: Infrastructure management
- Marketer: LinkedIn content creation and image generation
- Designer: Social media image design and generation
- Linter: Static analysis and code linting
- Automator: Desktop automation (macOS Shortcuts, AppleScript, Playwright)
- Scheduler: Calendar management (macOS Calendar, Google Calendar)
"""

from src.agents.sentinel import SentinelAgent
from src.agents.builder import BuilderAgent
from src.agents.inspector import InspectorAgent
from src.agents.watcher import WatcherAgent
from src.agents.scout import ScoutAgent
from src.agents.scribe import ScribeAgent
from src.agents.devops import DevOpsAgent
from src.agents.marketer import MarketerAgent
from src.agents.designer import DesignerAgent
from src.agents.linter import LinterAgent
from src.agents.automator import AutomatorAgent
from src.agents.scheduler import SchedulerAgent

__all__ = [
    "SentinelAgent",
    "BuilderAgent",
    "InspectorAgent",
    "WatcherAgent",
    "ScoutAgent",
    "ScribeAgent",
    "DevOpsAgent",
    "MarketerAgent",
    "DesignerAgent",
    "LinterAgent",
    "AutomatorAgent",
    "SchedulerAgent",
]
