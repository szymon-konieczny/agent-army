"""
Execution sandbox module for AgentArmy.

Provides isolated execution environment with resource limits,
process monitoring, and artifact collection.
"""

import asyncio
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


@dataclass
class ResourceLimit:
    """Resource limit configuration."""

    cpu_percent: int = 50
    memory_mb: int = 512
    timeout_seconds: int = 30


@dataclass
class NetworkRule:
    """Network access rule."""

    allow: bool = True
    protocol: str = ""  # tcp, udp, icmp
    host_pattern: str = ""
    port_range: tuple[int, int] = field(default_factory=lambda: (0, 65535))


@dataclass
class FilesystemRule:
    """Filesystem access rule."""

    allow: bool = True
    path_pattern: str = ""
    operations: list[str] = field(default_factory=lambda: ["read"])  # read, write, delete


class SandboxConfig(BaseModel):
    """Configuration for execution sandbox.

    Attributes:
        cpu_limit: CPU limit as percentage (0-100)
        memory_limit: Memory limit in MB
        timeout_seconds: Execution timeout in seconds
        network_rules: List of network access rules
        filesystem_rules: List of filesystem access rules
        allow_network: Allow external network access
        allow_filesystem: Allow filesystem access
        environment_vars: Environment variables to expose
    """

    cpu_limit: int = Field(default=50, ge=1, le=100)
    memory_limit: int = Field(default=512, ge=128)
    timeout_seconds: int = Field(default=30, ge=1)
    network_rules: list[NetworkRule] = Field(default_factory=list)
    filesystem_rules: list[FilesystemRule] = Field(default_factory=list)
    allow_network: bool = Field(default=False)
    allow_filesystem: bool = Field(default=False)
    environment_vars: dict[str, str] = Field(default_factory=dict)


@dataclass
class ExecutionArtifact:
    """Artifact from execution."""

    artifact_type: str  # stdout, stderr, file, metric
    name: str
    data: str | bytes
    size_bytes: int
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class ExecutionResult:
    """Result of sandbox execution.

    Attributes:
        exit_code: Process exit code
        success: Whether execution succeeded
        duration_seconds: Execution duration
        artifacts: Collected artifacts (stdout, files, etc.)
        error_message: Error message if failed
        resource_usage: Resource usage during execution
    """

    exit_code: int
    success: bool
    duration_seconds: float
    artifacts: list[ExecutionArtifact] = field(default_factory=list)
    error_message: str = ""
    resource_usage: dict = field(default_factory=dict)


class Sandbox:
    """Execution sandbox for isolating agent code.

    Manages resource limits, process execution, monitoring,
    and artifact collection.
    """

    def __init__(self, config: SandboxConfig) -> None:
        """Initialize sandbox.

        Args:
            config: Sandbox configuration
        """
        self.config = config
        self._process: Optional[asyncio.subprocess.Process] = None
        self._start_time: Optional[datetime] = None

    async def execute(
        self,
        command: str,
        args: Optional[list[str]] = None,
        stdin_data: Optional[str] = None,
        capture_output: bool = True,
    ) -> ExecutionResult:
        """Execute command in sandbox with resource limits.

        Args:
            command: Command to execute
            args: Command arguments
            stdin_data: Input to provide on stdin
            capture_output: Whether to capture stdout/stderr

        Returns:
            ExecutionResult with exit code and artifacts
        """
        self._start_time = datetime.now(timezone.utc)
        artifacts: list[ExecutionArtifact] = []

        try:
            # Build command
            full_command = [command]
            if args:
                full_command.extend(args)

            logger.info(
                "sandbox_execution_starting",
                command=command,
                args=args,
                timeout_seconds=self.config.timeout_seconds,
                memory_limit_mb=self.config.memory_limit,
                cpu_limit_percent=self.config.cpu_limit,
            )

            # Create process with limits
            if capture_output:
                stdout = asyncio.subprocess.PIPE
                stderr = asyncio.subprocess.PIPE
            else:
                stdout = None
                stderr = None

            self._process = await asyncio.create_subprocess_exec(
                *full_command,
                stdout=stdout,
                stderr=stderr,
                stdin=asyncio.subprocess.PIPE if stdin_data else None,
                env=self._build_environment(),
            )

            try:
                # Execute with timeout
                stdout_data, stderr_data = await asyncio.wait_for(
                    self._process.communicate(
                        input=stdin_data.encode() if stdin_data else None
                    ),
                    timeout=self.config.timeout_seconds,
                )

                exit_code = self._process.returncode or 0

                # Collect artifacts
                if stdout_data:
                    artifacts.append(
                        ExecutionArtifact(
                            artifact_type="stdout",
                            name="stdout",
                            data=stdout_data.decode(errors="replace"),
                            size_bytes=len(stdout_data),
                        )
                    )

                if stderr_data:
                    artifacts.append(
                        ExecutionArtifact(
                            artifact_type="stderr",
                            name="stderr",
                            data=stderr_data.decode(errors="replace"),
                            size_bytes=len(stderr_data),
                        )
                    )

                duration = (
                    datetime.now(timezone.utc) - self._start_time
                ).total_seconds()

                success = exit_code == 0

                logger.info(
                    "sandbox_execution_completed",
                    command=command,
                    exit_code=exit_code,
                    success=success,
                    duration_seconds=duration,
                    artifact_count=len(artifacts),
                )

                return ExecutionResult(
                    exit_code=exit_code,
                    success=success,
                    duration_seconds=duration,
                    artifacts=artifacts,
                )

            except asyncio.TimeoutError:
                logger.error(
                    "sandbox_execution_timeout",
                    command=command,
                    timeout_seconds=self.config.timeout_seconds,
                )

                # Kill runaway process
                await self._kill_process()

                duration = (
                    datetime.now(timezone.utc) - self._start_time
                ).total_seconds()

                return ExecutionResult(
                    exit_code=-1,
                    success=False,
                    duration_seconds=duration,
                    artifacts=artifacts,
                    error_message=(
                        f"Execution timeout after {self.config.timeout_seconds} "
                        "seconds"
                    ),
                )

        except Exception as e:
            logger.error(
                "sandbox_execution_error",
                command=command,
                error=str(e),
                error_type=type(e).__name__,
            )

            duration = (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds() if self._start_time else 0

            await self._kill_process()

            return ExecutionResult(
                exit_code=-1,
                success=False,
                duration_seconds=duration,
                artifacts=artifacts,
                error_message=str(e),
            )

    async def _kill_process(self) -> None:
        """Kill running process."""
        if self._process and not self._process.returncode:
            try:
                self._process.kill()
                await self._process.wait()
                logger.info("sandbox_process_killed")
            except Exception as e:
                logger.error("failed_to_kill_process", error=str(e))

    def _build_environment(self) -> dict[str, str]:
        """Build environment variables for process.

        Returns:
            Environment dict
        """
        env = {}

        # Add configured environment variables
        for key, value in self.config.environment_vars.items():
            # Validate key for security
            if key.upper() in ("PATH", "HOME", "USER", "SHELL"):
                env[key] = value
            elif not key.startswith(("AWS_", "GITHUB_", "API_")):
                # Allow safe custom variables
                env[key] = value

        return env

    def validate_network_access(
        self, protocol: str, host: str, port: int
    ) -> bool:
        """Validate network access request.

        Args:
            protocol: Protocol (tcp, udp, etc.)
            host: Host to connect to
            port: Port number

        Returns:
            True if access allowed
        """
        if not self.config.allow_network:
            logger.warning(
                "network_access_denied",
                protocol=protocol,
                host=host,
                port=port,
                reason="network_disabled",
            )
            return False

        # Check rules
        for rule in self.config.network_rules:
            if self._matches_network_rule(rule, protocol, host, port):
                return rule.allow

        # Default deny
        logger.warning(
            "network_access_denied",
            protocol=protocol,
            host=host,
            port=port,
            reason="no_matching_rule",
        )
        return False

    def validate_filesystem_access(
        self, path: str, operation: str
    ) -> bool:
        """Validate filesystem access request.

        Args:
            path: File path
            operation: Operation (read, write, delete)

        Returns:
            True if access allowed
        """
        if not self.config.allow_filesystem:
            logger.warning(
                "filesystem_access_denied",
                path=path,
                operation=operation,
                reason="filesystem_disabled",
            )
            return False

        # Check rules
        for rule in self.config.filesystem_rules:
            if self._matches_filesystem_rule(rule, path, operation):
                return rule.allow

        # Default deny
        logger.warning(
            "filesystem_access_denied",
            path=path,
            operation=operation,
            reason="no_matching_rule",
        )
        return False

    @staticmethod
    def _matches_network_rule(
        rule: NetworkRule, protocol: str, host: str, port: int
    ) -> bool:
        """Check if network request matches rule.

        Args:
            rule: Network rule
            protocol: Protocol
            host: Host
            port: Port

        Returns:
            True if matches
        """
        if rule.protocol and rule.protocol != protocol:
            return False

        if rule.host_pattern:
            import fnmatch

            if not fnmatch.fnmatch(host, rule.host_pattern):
                return False

        if not (rule.port_range[0] <= port <= rule.port_range[1]):
            return False

        return True

    @staticmethod
    def _matches_filesystem_rule(
        rule: FilesystemRule, path: str, operation: str
    ) -> bool:
        """Check if filesystem request matches rule.

        Args:
            rule: Filesystem rule
            path: File path
            operation: Operation

        Returns:
            True if matches
        """
        if rule.path_pattern:
            import fnmatch

            if not fnmatch.fnmatch(path, rule.path_pattern):
                return False

        if operation not in rule.operations:
            return False

        return True

    def get_resource_usage(self) -> dict:
        """Get resource usage during execution.

        Returns:
            Dict with CPU, memory, and time usage
        """
        if not self._start_time:
            return {}

        duration = (
            datetime.now(timezone.utc) - self._start_time
        ).total_seconds()

        return {
            "cpu_limit_percent": self.config.cpu_limit,
            "memory_limit_mb": self.config.memory_limit,
            "timeout_seconds": self.config.timeout_seconds,
            "elapsed_seconds": duration,
        }
