"""Static analysis and code linting agent — ESLint, Ruff, Prettier, Stylelint, etc."""

from datetime import datetime, timezone
from typing import Any

import structlog

from src.core.agent_base import AgentCapability, AgentIdentity, AgentState, BaseAgent

logger = structlog.get_logger(__name__)

# ── Linter tool presets ─────────────────────────────────────────
LINTER_TOOLS = {
    "eslint": {
        "label": "ESLint",
        "languages": ["javascript", "typescript", "jsx", "tsx"],
        "install": "npm install -D eslint",
        "run": "npx eslint . --format json",
        "fix": "npx eslint . --fix",
        "config_files": [".eslintrc", ".eslintrc.js", ".eslintrc.json", ".eslintrc.yml", "eslint.config.js", "eslint.config.mjs"],
        "description": "JavaScript/TypeScript linter — catches bugs, enforces code style, identifies anti-patterns.",
    },
    "prettier": {
        "label": "Prettier",
        "languages": ["javascript", "typescript", "css", "html", "json", "markdown", "yaml"],
        "install": "npm install -D prettier",
        "run": "npx prettier --check .",
        "fix": "npx prettier --write .",
        "config_files": [".prettierrc", ".prettierrc.js", ".prettierrc.json", "prettier.config.js"],
        "description": "Opinionated code formatter — consistent formatting across the codebase.",
    },
    "ruff": {
        "label": "Ruff",
        "languages": ["python"],
        "install": "pip install ruff",
        "run": "ruff check . --output-format json",
        "fix": "ruff check . --fix",
        "config_files": ["ruff.toml", "pyproject.toml"],
        "description": "Blazing-fast Python linter — replaces Flake8, isort, pycodestyle, and more.",
    },
    "pylint": {
        "label": "Pylint",
        "languages": ["python"],
        "install": "pip install pylint",
        "run": "pylint --output-format=json .",
        "fix": None,
        "config_files": [".pylintrc", "pyproject.toml"],
        "description": "Comprehensive Python static analysis — checks errors, style, refactoring opportunities.",
    },
    "mypy": {
        "label": "Mypy",
        "languages": ["python"],
        "install": "pip install mypy",
        "run": "mypy . --no-error-summary",
        "fix": None,
        "config_files": ["mypy.ini", "pyproject.toml", ".mypy.ini"],
        "description": "Python type checker — catches type errors before runtime.",
    },
    "stylelint": {
        "label": "Stylelint",
        "languages": ["css", "scss", "less"],
        "install": "npm install -D stylelint stylelint-config-standard",
        "run": "npx stylelint '**/*.{css,scss}' --formatter json",
        "fix": "npx stylelint '**/*.{css,scss}' --fix",
        "config_files": [".stylelintrc", ".stylelintrc.json", "stylelint.config.js"],
        "description": "CSS/SCSS linter — enforces consistent conventions and avoids errors.",
    },
    "rubocop": {
        "label": "RuboCop",
        "languages": ["ruby"],
        "install": "gem install rubocop",
        "run": "rubocop --format json",
        "fix": "rubocop --auto-correct",
        "config_files": [".rubocop.yml"],
        "description": "Ruby static analyzer and formatter.",
    },
    "golangci_lint": {
        "label": "golangci-lint",
        "languages": ["go"],
        "install": "go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest",
        "run": "golangci-lint run --out-format json",
        "fix": "golangci-lint run --fix",
        "config_files": [".golangci.yml", ".golangci.yaml"],
        "description": "Go linters aggregator — runs multiple Go linters in parallel.",
    },
    "clippy": {
        "label": "Clippy",
        "languages": ["rust"],
        "install": "rustup component add clippy",
        "run": "cargo clippy --message-format=json",
        "fix": "cargo clippy --fix --allow-dirty",
        "config_files": ["clippy.toml", ".clippy.toml"],
        "description": "Rust linter — catches common mistakes and improves code quality.",
    },
    "shellcheck": {
        "label": "ShellCheck",
        "languages": ["bash", "sh", "shell"],
        "install": "apt-get install shellcheck || brew install shellcheck",
        "run": "shellcheck -f json *.sh",
        "fix": None,
        "config_files": [".shellcheckrc"],
        "description": "Shell script analyzer — finds bugs, security issues, and style problems in bash/sh scripts.",
    },
    "hadolint": {
        "label": "Hadolint",
        "languages": ["dockerfile"],
        "install": "brew install hadolint || wget -O hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64",
        "run": "hadolint --format json Dockerfile",
        "fix": None,
        "config_files": [".hadolint.yaml"],
        "description": "Dockerfile linter — best practice validation for container builds.",
    },
}

# File extension → language mapping for auto-detection
EXT_TO_LANGUAGE = {
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript",
    ".py": "python", ".pyi": "python",
    ".css": "css", ".scss": "scss", ".less": "less",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".html": "html", ".htm": "html",
    ".json": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown",
}


class LinterAgent(BaseAgent):
    """Static analysis and code linting agent.

    Responsibilities:
    - Runs linters (ESLint, Ruff, Prettier, Stylelint, Mypy, etc.)
    - Auto-detects the project's language stack and applicable tools
    - Auto-fixes issues where supported
    - Provides lint reports with severity, location, and fix suggestions
    - Sets up linter configs for new projects
    - Enforces code style consistency across the codebase

    Capabilities:
    - lint_project: Run all applicable linters on the project
    - lint_file: Run linters on a specific file or directory
    - fix_issues: Auto-fix all fixable lint issues
    - setup_linter: Initialize a linter with a recommended config
    - lint_report: Generate a comprehensive code quality report
    """

    def __init__(
        self,
        agent_id: str = "linter-001",
        name: str = "Linter Agent",
        role: str = "linter",
    ) -> None:
        """Initialize the Linter agent.

        Args:
            agent_id: Unique agent identifier.
            name: Display name for the agent.
            role: Agent role classification.
        """
        identity = AgentIdentity(
            id=agent_id,
            name=name,
            role=role,
            security_level=2,
            capabilities=[
                AgentCapability(
                    name="lint_project",
                    version="1.0.0",
                    description="Run all applicable linters on the project",
                    parameters={
                        "tools": "list[str]",
                        "auto_detect": "bool",
                    },
                ),
                AgentCapability(
                    name="lint_file",
                    version="1.0.0",
                    description="Run linters on specific files or directories",
                    parameters={
                        "path": "str",
                        "tool": "str",
                    },
                ),
                AgentCapability(
                    name="fix_issues",
                    version="1.0.0",
                    description="Auto-fix all fixable lint issues",
                    parameters={
                        "tools": "list[str]",
                        "dry_run": "bool",
                    },
                ),
                AgentCapability(
                    name="setup_linter",
                    version="1.0.0",
                    description="Initialize a linter with recommended config",
                    parameters={
                        "tool": "str",
                        "preset": "str",
                    },
                ),
                AgentCapability(
                    name="lint_report",
                    version="1.0.0",
                    description="Generate a comprehensive code quality report",
                    parameters={
                        "format": "str",
                        "include_summary": "bool",
                    },
                ),
            ],
        )
        super().__init__(identity)
        self._lint_history: list[dict[str, Any]] = []

    async def startup(self) -> None:
        """Initialize the Linter agent.

        Raises:
            Exception: If startup fails.
        """
        await super().startup()
        await logger.ainfo("linter_startup", agent_id=self.identity.id)

    async def shutdown(self) -> None:
        """Shutdown the Linter agent gracefully.

        Raises:
            Exception: If shutdown fails.
        """
        await logger.ainfo(
            "linter_shutdown",
            agent_id=self.identity.id,
            lint_runs=len(self._lint_history),
        )
        await super().shutdown()

    def _get_system_context(self) -> str:
        """Get system context for LLM reasoning.

        Returns:
            System context string describing the agent's role and expertise.
        """
        tool_list = "\n".join(
            f"  - **{info['label']}** ({', '.join(info['languages'])}): {info['description']}\n"
            f"    Install: `{info['install']}` | Run: `{info['run']}`"
            + (f" | Fix: `{info['fix']}`" if info.get("fix") else "")
            for key, info in LINTER_TOOLS.items()
        )

        return (
            "You are a code quality and static analysis expert. "
            "You specialize in running linters, formatters, and type checkers "
            "to find bugs, enforce coding standards, and improve code quality.\n\n"
            "KEY RESPONSIBILITIES:\n"
            "- Detect the project's language stack by examining files and configs\n"
            "- Run the appropriate linters for each language found\n"
            "- Parse lint output and present issues clearly (file, line, severity, message)\n"
            "- Auto-fix issues where the tool supports it\n"
            "- Help set up linter configs for new projects\n"
            "- Explain why specific rules matter and how to resolve violations\n\n"
            "AVAILABLE LINTER TOOLS:\n"
            f"{tool_list}\n\n"
            "AUTO-DETECTION STRATEGY:\n"
            "1. Check for existing config files (.eslintrc, pyproject.toml, etc.)\n"
            "2. Examine file extensions to determine languages in use\n"
            "3. Check package.json for JS/TS projects, pyproject.toml/setup.py for Python\n"
            "4. Check for Cargo.toml (Rust), go.mod (Go), Gemfile (Ruby)\n"
            "5. Run each detected tool and aggregate results\n\n"
            "OUTPUT FORMAT:\n"
            "- Always present results grouped by severity: errors first, then warnings, then info\n"
            "- Include the file path, line number, rule ID, and message for each issue\n"
            "- At the end, provide a summary: total issues, errors, warnings, fixable count\n"
            "- If auto-fix is available, offer to run it\n\n"
            "WHEN SETTING UP LINTERS:\n"
            "- Recommend sensible defaults (e.g., eslint with recommended + prettier)\n"
            "- Create the config file and install dependencies\n"
            "- Add lint/format scripts to package.json or Makefile\n"
            "- Suggest pre-commit hooks for automated enforcement\n"
        )

    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process lint-related tasks.

        Supported task types:
        - lint_project: Run all applicable linters
        - lint_file: Lint specific files
        - fix_issues: Auto-fix lint issues
        - setup_linter: Set up a linter tool
        - lint_report: Generate quality report

        Args:
            task: Task payload with type and parameters.

        Returns:
            Result dictionary with lint findings.

        Raises:
            ValueError: If task type is unsupported.
        """
        task_type = task.get("type", "unknown")
        task_id = task.get("id", "unknown")

        await logger.ainfo(
            "linter_processing_task",
            task_id=task_id,
            task_type=task_type,
        )

        # ── Chain-of-Thought reasoning ────────────────────────────
        reasoning = await self.think(task)
        task.setdefault("context", {})["_reasoning"] = reasoning.conclusion

        try:
            if task_type == "lint_project":
                result = await self._handle_lint_project(task)
            elif task_type == "lint_file":
                result = await self._handle_lint_file(task)
            elif task_type == "fix_issues":
                result = await self._handle_fix_issues(task)
            elif task_type == "setup_linter":
                result = await self._handle_setup_linter(task)
            elif task_type == "lint_report":
                result = await self._handle_lint_report(task)
            else:
                return await self._handle_chat_message(task)
            result["reasoning"] = reasoning.to_audit_dict()
            return result
        except Exception as exc:
            await logger.aerror(
                "linter_task_error",
                task_id=task_id,
                error=str(exc),
            )
            raise

    async def _handle_lint_project(self, task: dict[str, Any]) -> dict[str, Any]:
        """Run all applicable linters on the project.

        Args:
            task: Task with lint parameters.

        Returns:
            Dictionary with lint results.
        """
        params = task.get("context", {})
        tools = params.get("tools", [])
        auto_detect = params.get("auto_detect", True)

        await logger.ainfo(
            "lint_project_started",
            tools=tools,
            auto_detect=auto_detect,
        )

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            result = {
                "status": "completed",
                "type": "lint_project",
                "analysis": analysis,
                "auto_detect": auto_detect,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self._lint_history.append(result)
            return result
        except Exception as exc:
            await logger.awarning("lint_project_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "lint_project",
                "summary": "Run linters on the project — LLM unavailable for detailed analysis.",
                "available_tools": list(LINTER_TOOLS.keys()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_lint_file(self, task: dict[str, Any]) -> dict[str, Any]:
        """Run linters on a specific file or directory.

        Args:
            task: Task with file path and tool parameters.

        Returns:
            Dictionary with lint results for the file.
        """
        params = task.get("context", {})
        path = params.get("path", ".")
        tool = params.get("tool", "auto")

        await logger.ainfo("lint_file_started", path=path, tool=tool)

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            return {
                "status": "completed",
                "type": "lint_file",
                "path": path,
                "tool": tool,
                "analysis": analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("lint_file_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "lint_file",
                "path": path,
                "tool": tool,
                "summary": f"Lint {path} — LLM unavailable for detailed analysis.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_fix_issues(self, task: dict[str, Any]) -> dict[str, Any]:
        """Auto-fix all fixable lint issues.

        Args:
            task: Task with fix parameters.

        Returns:
            Dictionary with fix results.
        """
        params = task.get("context", {})
        tools = params.get("tools", [])
        dry_run = params.get("dry_run", False)

        await logger.ainfo("lint_fix_started", tools=tools, dry_run=dry_run)

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            return {
                "status": "completed",
                "type": "fix_issues",
                "tools": tools,
                "dry_run": dry_run,
                "analysis": analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("lint_fix_llm_fallback", error=str(exc))
            fixable_tools = [
                key for key, info in LINTER_TOOLS.items()
                if info.get("fix") is not None
            ]
            return {
                "status": "completed",
                "type": "fix_issues",
                "tools_with_autofix": fixable_tools,
                "summary": "Auto-fix available — LLM unavailable for detailed analysis.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_setup_linter(self, task: dict[str, Any]) -> dict[str, Any]:
        """Set up a linter with recommended configuration.

        Args:
            task: Task with setup parameters.

        Returns:
            Dictionary with setup instructions.
        """
        params = task.get("context", {})
        tool = params.get("tool", "eslint")
        preset = params.get("preset", "recommended")

        await logger.ainfo("linter_setup_started", tool=tool, preset=preset)

        tool_info = LINTER_TOOLS.get(tool)

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            return {
                "status": "completed",
                "type": "setup_linter",
                "tool": tool,
                "preset": preset,
                "tool_info": tool_info,
                "analysis": analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("linter_setup_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "setup_linter",
                "tool": tool,
                "preset": preset,
                "tool_info": tool_info,
                "summary": f"Set up {tool_info['label'] if tool_info else tool} — see install command above.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _handle_lint_report(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate a comprehensive lint quality report.

        Args:
            task: Task with report parameters.

        Returns:
            Dictionary with quality report.
        """
        params = task.get("context", {})
        report_format = params.get("format", "markdown")
        include_summary = params.get("include_summary", True)

        await logger.ainfo(
            "lint_report_started",
            report_format=report_format,
            include_summary=include_summary,
        )

        try:
            chain = await self.think(task, strategy="step_by_step")
            analysis = chain.conclusion

            return {
                "status": "completed",
                "type": "lint_report",
                "report_format": report_format,
                "analysis": analysis,
                "reasoning_steps": len(chain.steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            await logger.awarning("lint_report_llm_fallback", error=str(exc))
            return {
                "status": "completed",
                "type": "lint_report",
                "report_format": report_format,
                "available_tools": {
                    key: {
                        "label": info["label"],
                        "languages": info["languages"],
                    }
                    for key, info in LINTER_TOOLS.items()
                },
                "lint_history_count": len(self._lint_history),
                "summary": "Code quality report — LLM unavailable for detailed analysis.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
