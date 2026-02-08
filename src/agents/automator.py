"""Desktop automation agent — macOS Shortcuts, AppleScript, Playwright browser RPA."""

import json
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from src.core.agent_base import AgentCapability, AgentIdentity, AgentState, BaseAgent
from src.platform.base import DesktopAdapter, PlatformCapability

logger = structlog.get_logger(__name__)

# ── macOS Shortcuts CLI reference ───────────────────────────────
SHORTCUTS_CLI = {
    "list": "shortcuts list",
    "run": 'shortcuts run "{name}"',
    "run_with_input": 'shortcuts run "{name}" --input-path "{input_path}"',
    "run_with_output": 'shortcuts run "{name}" --output-path "{output_path}"',
    "run_stdin": 'echo "{input}" | shortcuts run "{name}"',
    "view": 'shortcuts view "{name}"',
    "sign": 'shortcuts sign --mode anyone --input "{file}" --output "{output}"',
}

# ── AppleScript / osascript snippets ────────────────────────────
APPLESCRIPT_TEMPLATES = {
    "get_calendar_events": {
        "label": "Get Calendar Events (today)",
        "description": "Reads today's events from macOS Calendar.app",
        "script": '''
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
    },
    "get_calendar_events_range": {
        "label": "Get Calendar Events (date range)",
        "description": "Reads events in a date range from macOS Calendar.app",
        "script": '''
-- Pass startDate and endDate as "YYYY-MM-DD" strings
on run argv
    set startStr to item 1 of argv
    set endStr to item 2 of argv
    tell application "Calendar"
        set output to ""
        repeat with cal in calendars
            set evts to (every event of cal whose start date >= date startStr and start date <= date endStr)
            repeat with evt in evts
                set output to output & (summary of evt) & " | " & (start date of evt) & " - " & (end date of evt) & " | " & (name of cal) & linefeed
            end repeat
        end repeat
        return output
    end tell
end run
''',
    },
    "get_reminders": {
        "label": "Get Reminders",
        "description": "Reads incomplete reminders from macOS Reminders.app",
        "script": '''
tell application "Reminders"
    set output to ""
    repeat with reminderList in lists
        set incompleteItems to (every reminder of reminderList whose completed is false)
        repeat with r in incompleteItems
            set output to output & (name of reminderList) & ": " & (name of r)
            if due date of r is not missing value then
                set output to output & " (due: " & (due date of r) & ")"
            end if
            set output to output & linefeed
        end repeat
    end repeat
    return output
end tell
''',
    },
    "create_reminder": {
        "label": "Create Reminder",
        "description": "Creates a new reminder in macOS Reminders.app",
        "script": '''
on run argv
    set reminderTitle to item 1 of argv
    tell application "Reminders"
        tell list "Reminders"
            make new reminder with properties {{name:reminderTitle}}
        end tell
    end tell
    return "Created: " & reminderTitle
end run
''',
    },
    "get_frontmost_app": {
        "label": "Get Frontmost App",
        "description": "Returns the name and window title of the frontmost application",
        "script": '''
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
    },
    "list_open_apps": {
        "label": "List Open Applications",
        "description": "Lists all running applications",
        "script": '''
tell application "System Events"
    set appList to ""
    repeat with proc in (every application process whose visible is true)
        set appList to appList & (name of proc) & linefeed
    end repeat
    return appList
end tell
''',
    },
    "open_url_in_browser": {
        "label": "Open URL in Default Browser",
        "description": "Opens a URL in the system default browser",
        "script": '''
on run argv
    set targetURL to item 1 of argv
    open location targetURL
end run
''',
    },
    "notification": {
        "label": "Show Notification",
        "description": "Displays a macOS notification",
        "script": '''
on run argv
    set notifTitle to item 1 of argv
    set notifBody to item 2 of argv
    display notification notifBody with title notifTitle
end run
''',
    },
    "clipboard_get": {
        "label": "Get Clipboard",
        "description": "Returns the current clipboard text content",
        "script": '''
return (the clipboard as text)
''',
    },
    "clipboard_set": {
        "label": "Set Clipboard",
        "description": "Sets the clipboard text content",
        "script": '''
on run argv
    set the clipboard to (item 1 of argv)
    return "Clipboard set"
end run
''',
    },
}

# ── Playwright browser automation presets ────────────────────────
BROWSER_PRESETS = {
    "jira_timesheet": {
        "label": "Jira/Tempo Timesheet",
        "description": "Log hours in Jira Tempo timesheet based on calendar events",
        "steps": [
            "Read calendar events for the target date",
            "Navigate to Tempo timesheet page",
            "For each event, create a worklog entry",
            "Map event titles to Jira issue keys where possible",
            "Fill hours, description, and submit",
        ],
    },
    "web_scrape": {
        "label": "Web Scrape",
        "description": "Extract structured data from a web page",
        "steps": [
            "Navigate to the target URL",
            "Wait for page load and dynamic content",
            "Extract data using CSS selectors or XPath",
            "Return structured JSON result",
        ],
    },
    "form_fill": {
        "label": "Form Fill",
        "description": "Automatically fill a web form with provided data",
        "steps": [
            "Navigate to the form URL",
            "Identify form fields by label or selector",
            "Fill each field with provided values",
            "Optionally submit the form",
        ],
    },
    "screenshot_page": {
        "label": "Screenshot Page",
        "description": "Take a full-page screenshot of a URL",
        "steps": [
            "Navigate to the URL",
            "Wait for page load",
            "Take full-page screenshot",
            "Save to project directory",
        ],
    },
    "monitor_page": {
        "label": "Monitor Page Changes",
        "description": "Check a page periodically for content changes",
        "steps": [
            "Navigate to the target URL",
            "Extract the monitored content area",
            "Compare with previous snapshot",
            "Alert if changes detected",
        ],
    },
}

# ── Common automation workflow templates ─────────────────────────
WORKFLOW_TEMPLATES = {
    "calendar_to_timesheet": {
        "label": "Calendar → Timesheet",
        "description": (
            "Reads today's calendar events via macOS Calendar.app (AppleScript), "
            "maps them to Jira issues, and logs hours in Tempo."
        ),
        "steps": [
            {"tool": "applescript", "action": "get_calendar_events", "description": "Read today's events"},
            {"tool": "llm", "action": "parse_events", "description": "Parse events, extract Jira issue keys from titles"},
            {"tool": "playwright", "action": "jira_timesheet", "description": "Log hours in Tempo for each event"},
        ],
    },
    "daily_standup_prep": {
        "label": "Daily Standup Prep",
        "description": (
            "Reads yesterday's calendar, today's schedule, open reminders, "
            "and generates a standup summary."
        ),
        "steps": [
            {"tool": "applescript", "action": "get_calendar_events", "description": "Read today's events"},
            {"tool": "applescript", "action": "get_reminders", "description": "Get open reminders/tasks"},
            {"tool": "llm", "action": "summarize", "description": "Generate standup notes"},
        ],
    },
    "expense_report": {
        "label": "Receipt → Expense Report",
        "description": (
            "Takes receipt images/PDFs, extracts amounts via OCR, "
            "and fills an expense report form."
        ),
        "steps": [
            {"tool": "llm", "action": "ocr", "description": "Extract receipt data (vendor, amount, date)"},
            {"tool": "playwright", "action": "form_fill", "description": "Fill expense report form"},
        ],
    },
    "shortcut_chain": {
        "label": "Shortcut Chain",
        "description": (
            "Run multiple macOS Shortcuts in sequence, piping output from one to the next."
        ),
        "steps": [
            {"tool": "shortcuts", "action": "run", "description": "Run first shortcut"},
            {"tool": "shortcuts", "action": "run_with_input", "description": "Pipe result to next shortcut"},
        ],
    },
}


class AutomatorAgent(BaseAgent):
    """Desktop automation agent combining macOS Shortcuts, AppleScript, and Playwright.

    Responsibilities:
    - Discovers and runs macOS Shortcuts from the command line
    - Executes AppleScript to interact with native macOS apps (Calendar, Reminders, Finder, etc.)
    - Uses Playwright for browser-based RPA (Jira/Tempo timesheets, form filling, scraping)
    - Chains multiple automation tools into workflows (e.g., Calendar → Tempo timesheet)
    - Monitors pages for changes, takes screenshots, fills forms
    - Provides workflow templates for common repetitive tasks

    Capabilities:
    - run_shortcut: Discover and run macOS Shortcuts
    - run_applescript: Execute AppleScript for native app automation
    - browser_automate: Playwright-based browser automation
    - run_workflow: Execute a multi-step automation workflow
    - list_capabilities: Show available automations and templates
    """

    def __init__(
        self,
        agent_id: str = "automator-001",
        name: str = "Automator Agent",
        role: str = "automator",
        desktop: Optional[DesktopAdapter] = None,
    ) -> None:
        """Initialize the Automator agent.

        Args:
            agent_id: Unique agent identifier.
            name: Display name for the agent.
            role: Agent role classification.
            desktop: Platform-specific desktop adapter (auto-detected if None).
        """
        self._desktop = desktop
        identity = AgentIdentity(
            id=agent_id,
            name=name,
            role=role,
            security_level=3,  # Higher security — runs system commands
            capabilities=[
                AgentCapability(
                    name="run_shortcut",
                    version="1.0.0",
                    description="Discover and run macOS Shortcuts",
                    parameters={
                        "name": "str",
                        "input": "str",
                        "output_path": "str",
                    },
                ),
                AgentCapability(
                    name="run_applescript",
                    version="1.0.0",
                    description="Execute AppleScript for native macOS app automation",
                    parameters={
                        "template": "str",
                        "script": "str",
                        "args": "list[str]",
                    },
                ),
                AgentCapability(
                    name="browser_automate",
                    version="1.0.0",
                    description="Playwright-based browser automation (forms, scraping, RPA)",
                    parameters={
                        "preset": "str",
                        "url": "str",
                        "actions": "list[dict]",
                    },
                ),
                AgentCapability(
                    name="run_workflow",
                    version="1.0.0",
                    description="Execute a multi-step automation workflow",
                    parameters={
                        "template": "str",
                        "params": "dict",
                    },
                ),
                AgentCapability(
                    name="list_capabilities",
                    version="1.0.0",
                    description="Show all available automations, shortcuts, and templates",
                    parameters={},
                ),
            ],
        )
        super().__init__(identity)
        self._automation_history: list[dict[str, Any]] = []

    async def startup(self) -> None:
        """Initialize the Automator agent.

        Raises:
            Exception: If startup fails.
        """
        await super().startup()
        await logger.ainfo("automator_startup", agent_id=self.identity.id)

    async def shutdown(self) -> None:
        """Shutdown the Automator agent gracefully.

        Raises:
            Exception: If shutdown fails.
        """
        await logger.ainfo(
            "automator_shutdown",
            agent_id=self.identity.id,
            automations_run=len(self._automation_history),
        )
        await super().shutdown()

    @property
    def desktop(self) -> DesktopAdapter:
        """Lazy-load the desktop adapter if not injected."""
        if self._desktop is None:
            from src.platform import get_desktop_adapter
            self._desktop = get_desktop_adapter()
        return self._desktop

    def _get_system_context(self) -> str:
        """Get system context for LLM reasoning.

        Dynamically adjusts the prompt based on the current platform's
        capabilities — AppleScript/Shortcuts sections are only shown when
        the platform supports them.

        Returns:
            System context string describing the agent's role and expertise.
        """
        caps = {c.name: c for c in self.desktop.capabilities()}

        sections: list[str] = [
            "You are a desktop automation expert specializing in cross-platform "
            "automation, browser RPA, and workflow orchestration.\n"
        ]

        # ── Shortcuts (macOS only) ──
        if caps.get("shortcuts", PlatformCapability("shortcuts", False)).available:
            shortcuts_ref = "\n".join(
                f"  - {label}: `{cmd}`"
                for label, cmd in SHORTCUTS_CLI.items()
            )
            sections.append(
                "## SHORTCUTS CLI\n"
                "Shortcuts can be run from the terminal (macOS Monterey+):\n"
                f"{shortcuts_ref}\n"
                "To discover available shortcuts, run: `shortcuts list`\n"
                "Shortcuts can accept input via stdin or file and produce output.\n"
                "Always list available shortcuts first before running one.\n"
            )

        # ── AppleScript (macOS only) ──
        if caps.get("applescript", PlatformCapability("applescript", False)).available:
            applescript_ref = "\n".join(
                f"  - **{info['label']}**: {info['description']}"
                for key, info in APPLESCRIPT_TEMPLATES.items()
            )
            sections.append(
                "## APPLESCRIPT (osascript)\n"
                "You can control native macOS apps via `osascript`:\n"
                "  - `osascript -e 'tell application \"Calendar\" to ...'`\n"
                "  - `osascript script.scpt arg1 arg2`\n"
                "Available templates:\n"
                f"{applescript_ref}\n"
                "To run multi-line AppleScript, write it to a temp .scpt file first,\n"
                "then execute with `osascript /tmp/script.scpt`.\n"
            )

        # ── PowerShell (Windows only) ──
        if caps.get("powershell", PlatformCapability("powershell", False)).available:
            sections.append(
                "## POWERSHELL\n"
                "You can automate Windows desktop tasks via PowerShell:\n"
                "  - `Get-Clipboard` / `Set-Clipboard` for clipboard\n"
                "  - `Get-Process` to list running apps\n"
                "  - Toast notifications via Windows.UI.Notifications\n"
                "  - Task Scheduler for recurring automation\n"
            )

        # ── Bash (Linux / macOS) ──
        if caps.get("bash", PlatformCapability("bash", False)).available:
            sections.append(
                "## BASH SCRIPTING\n"
                "You can run bash scripts natively for automation.\n"
                "Clipboard: xclip / xsel / wl-clipboard\n"
                "Notifications: notify-send\n"
                "Window info: xdotool / wmctrl\n"
            )

        # ── Browser presets (always available — Playwright is cross-platform) ──
        browser_ref = "\n".join(
            f"  - **{info['label']}**: {info['description']}"
            for key, info in BROWSER_PRESETS.items()
        )
        sections.append(
            "## BROWSER AUTOMATION (Playwright)\n"
            "For web-based automation, use the built-in Playwright runner:\n"
            f"{browser_ref}\n"
            "The Playwright runner is accessible via the `/test/e2e/smoke` endpoint\n"
            "for quick page checks, or write full Playwright scripts in TypeScript/Python.\n"
        )

        # ── Workflow templates (always shown) ──
        workflow_ref = "\n".join(
            f"  - **{info['label']}**: {info['description']}"
            for key, info in WORKFLOW_TEMPLATES.items()
        )
        sections.append(
            "## WORKFLOW TEMPLATES\n"
            "Pre-built multi-step workflows:\n"
            f"{workflow_ref}\n"
        )

        # ── Capabilities summary ──
        cap_lines = "\n".join(
            f"  - {c.name}: {'✓' if c.available else '✗'}"
            + (f" ({c.reason})" if c.reason else "")
            for c in self.desktop.capabilities()
        )
        sections.append(
            "## PLATFORM CAPABILITIES\n"
            f"{cap_lines}\n"
        )

        sections.append(
            "## EXECUTION STRATEGY\n"
            "1. **Discover**: List available shortcuts, check which apps are running\n"
            "2. **Plan**: Break the task into steps using the right tool for each\n"
            "3. **Execute**: Run each step, capture output, pass to the next step\n"
            "4. **Verify**: Check results and report back to the user\n\n"
            "## SAFETY RULES\n"
            "- Always show the user what you plan to do before executing\n"
            "- Never delete files, emails, or data without explicit confirmation\n"
            "- Never submit forms with financial data\n"
            "- Ask for confirmation before running destructive operations\n"
            "- Log all automation actions for audit trail\n"
        )

        return "\n\n".join(sections)

    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process automation-related tasks.

        Supported task types:
        - run_shortcut: Discover and run macOS Shortcuts
        - run_applescript: Execute AppleScript templates
        - browser_automate: Browser-based automation
        - run_workflow: Multi-step workflow execution
        - list_capabilities: Show available automations

        Args:
            task: Task payload with type and parameters.

        Returns:
            Result dictionary with automation output.

        Raises:
            ValueError: If task type is unsupported.
        """
        task_type = task.get("type", "unknown")
        task_id = task.get("id", "unknown")

        await logger.ainfo(
            "automator_processing_task",
            task_id=task_id,
            task_type=task_type,
        )

        # ── Chain-of-Thought reasoning ────────────────────────────
        reasoning = await self.think(task)
        task.setdefault("context", {})["_reasoning"] = reasoning.conclusion

        try:
            if task_type == "run_shortcut":
                result = await self._handle_run_shortcut(task)
            elif task_type == "run_applescript":
                result = await self._handle_run_applescript(task)
            elif task_type == "browser_automate":
                result = await self._handle_browser_automate(task)
            elif task_type == "run_workflow":
                result = await self._handle_run_workflow(task)
            elif task_type == "list_capabilities":
                result = await self._handle_list_capabilities(task)
            else:
                return await self._handle_chat_message(task)
            result["reasoning"] = reasoning.to_audit_dict()
            return result
        except Exception as exc:
            await logger.aerror(
                "automator_task_error",
                task_id=task_id,
                error=str(exc),
            )
            raise

    async def _handle_run_shortcut(self, task: dict[str, Any]) -> dict[str, Any]:
        """Run a macOS Shortcut by name.

        Args:
            task: Task with shortcut parameters.

        Returns:
            Dictionary with execution result.
        """
        params = task.get("context", {})
        name = params.get("name", "")
        input_data = params.get("input", "")

        await logger.ainfo("shortcut_run", name=name)

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            self._automation_history.append({
                "type": "shortcut",
                "name": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            return {
                "status": "completed",
                "type": "run_shortcut",
                "shortcut_name": name,
                "analysis": analysis,
                "cli_reference": SHORTCUTS_CLI,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("shortcut_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "run_shortcut",
                "shortcut_name": name,
                "cli_reference": SHORTCUTS_CLI,
                "summary": f"Run shortcut '{name}' — use `shortcuts run \"{name}\"`",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_run_applescript(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute an AppleScript template or custom script.

        Args:
            task: Task with AppleScript parameters.

        Returns:
            Dictionary with execution result.
        """
        params = task.get("context", {})
        template_name = params.get("template", "")
        custom_script = params.get("script", "")
        args = params.get("args", [])

        await logger.ainfo("applescript_run", template=template_name)

        template = APPLESCRIPT_TEMPLATES.get(template_name)

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            self._automation_history.append({
                "type": "applescript",
                "template": template_name or "custom",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            return {
                "status": "completed",
                "type": "run_applescript",
                "template": template_name,
                "template_info": template,
                "analysis": analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("applescript_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "run_applescript",
                "template": template_name,
                "template_info": template,
                "available_templates": list(APPLESCRIPT_TEMPLATES.keys()),
                "summary": "AppleScript execution — LLM unavailable for detailed analysis.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_browser_automate(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute browser automation via Playwright.

        Args:
            task: Task with browser automation parameters.

        Returns:
            Dictionary with automation result.
        """
        params = task.get("context", {})
        preset = params.get("preset", "")
        url = params.get("url", "")

        await logger.ainfo("browser_automate", preset=preset, url=url)

        preset_info = BROWSER_PRESETS.get(preset)

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            self._automation_history.append({
                "type": "browser",
                "preset": preset,
                "url": url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            return {
                "status": "completed",
                "type": "browser_automate",
                "preset": preset,
                "url": url,
                "preset_info": preset_info,
                "analysis": analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("browser_automate_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "browser_automate",
                "preset": preset,
                "url": url,
                "preset_info": preset_info,
                "available_presets": list(BROWSER_PRESETS.keys()),
                "summary": "Browser automation — LLM unavailable for detailed analysis.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_run_workflow(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a multi-step automation workflow.

        Args:
            task: Task with workflow parameters.

        Returns:
            Dictionary with workflow execution result.
        """
        params = task.get("context", {})
        template_name = params.get("template", "")
        workflow_params = params.get("params", {})

        await logger.ainfo("workflow_run", template=template_name)

        template = WORKFLOW_TEMPLATES.get(template_name)

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            self._automation_history.append({
                "type": "workflow",
                "template": template_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            return {
                "status": "completed",
                "type": "run_workflow",
                "template": template_name,
                "template_info": template,
                "analysis": analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("workflow_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "run_workflow",
                "template": template_name,
                "template_info": template,
                "available_workflows": list(WORKFLOW_TEMPLATES.keys()),
                "summary": "Workflow execution — LLM unavailable for detailed analysis.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_list_capabilities(self, task: dict[str, Any]) -> dict[str, Any]:
        """List all available automation capabilities.

        Args:
            task: Task (no special parameters needed).

        Returns:
            Dictionary with all available automations.
        """
        return {
            "status": "completed",
            "type": "list_capabilities",
            "shortcuts_cli": SHORTCUTS_CLI,
            "applescript_templates": {
                key: {"label": info["label"], "description": info["description"]}
                for key, info in APPLESCRIPT_TEMPLATES.items()
            },
            "browser_presets": {
                key: {"label": info["label"], "description": info["description"]}
                for key, info in BROWSER_PRESETS.items()
            },
            "workflow_templates": {
                key: {"label": info["label"], "description": info["description"]}
                for key, info in WORKFLOW_TEMPLATES.items()
            },
            "automation_history_count": len(self._automation_history),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
