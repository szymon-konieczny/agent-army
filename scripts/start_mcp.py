#!/usr/bin/env python3
"""Start the AgentArmy MCP server.

This script launches the AgentArmy MCP (Model Context Protocol) server,
making AgentArmy capabilities available to external AI tools.

Usage:
    python scripts/start_mcp.py [--transport stdio|sse] [--port 8001]

    # Run with stdio transport (default, for Claude Desktop/Code):
    python scripts/start_mcp.py

    # Run with SSE transport (HTTP):
    python scripts/start_mcp.py --transport sse --port 8001

    # Run with custom API endpoint:
    python scripts/start_mcp.py --api-url http://api.example.com:8000

Configuration:
    The server reads configuration from config/mcp_server.yaml
    Environment variables override YAML configuration:
      - AGENTARMY_MCP_API_URL: API base URL
      - AGENTARMY_MCP_TRANSPORT: stdio or sse
      - AGENTARMY_MCP_SSE_PORT: SSE port
      - AGENTARMY_MCP_SSE_HOST: SSE bind address
      - AGENTARMY_NEO4J_PASSWORD: Neo4j password
      - AGENTARMY_REDIS_URL: Redis connection URL
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import structlog
import yaml
from src.mcp_server.server import MCPServer


def load_config(config_path: str = "config/mcp_server.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        # Use defaults if config doesn't exist
        return {
            "server": {
                "transport": {
                    "stdio": {"enabled": True},
                    "sse": {"enabled": True, "host": "0.0.0.0", "port": 8001},
                }
            },
            "api": {"base_url": "http://localhost:8000", "timeout": 30},
        }

    with open(config_file) as f:
        return yaml.safe_load(f) or {}


def get_env_override(key: str, default: str) -> str:
    """Get configuration value with environment variable override.

    Args:
        key: Environment variable name
        default: Default value if env var not set

    Returns:
        Configuration value
    """
    return os.getenv(key, default)


async def main() -> None:
    """Main entry point."""
    # Setup logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger(__name__)

    # Parse command line arguments
    transport = "stdio"
    sse_port = 8001
    sse_host = "0.0.0.0"
    api_url = "http://localhost:8000"

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--transport" and i < len(sys.argv) - 1:
            transport = sys.argv[i + 1]
        elif arg == "--port" and i < len(sys.argv) - 1:
            sse_port = int(sys.argv[i + 1])
        elif arg == "--host" and i < len(sys.argv) - 1:
            sse_host = sys.argv[i + 1]
        elif arg == "--api-url" and i < len(sys.argv) - 1:
            api_url = sys.argv[i + 1]
        elif arg in ("--help", "-h"):
            print(__doc__)
            return

    # Load configuration
    config = load_config()

    # Apply environment overrides
    api_url = get_env_override("AGENTARMY_MCP_API_URL", api_url)
    transport = get_env_override("AGENTARMY_MCP_TRANSPORT", transport)
    sse_port = int(get_env_override("AGENTARMY_MCP_SSE_PORT", str(sse_port)))
    sse_host = get_env_override("AGENTARMY_MCP_SSE_HOST", sse_host)

    logger.info(
        "starting_agentarmy_mcp",
        transport=transport,
        api_url=api_url,
        sse_port=sse_port if transport == "sse" else None,
    )

    try:
        # Create and run server
        server = MCPServer(
            api_base_url=api_url,
            api_timeout=config.get("api", {}).get("timeout", 30),
            redis_url=os.getenv("AGENTARMY_REDIS_URL"),
            neo4j_uri=config.get("neo4j", {}).get("uri"),
        )

        if transport == "stdio":
            logger.info("mcp_server_ready", transport="stdio")
            logger.info(
                "info",
                message="Add to Claude Desktop config or use with Claude Code",
                config_path="~/.config/Claude/claude_desktop_config.json",
            )
            await server.run_stdio()

        elif transport == "sse":
            logger.info("mcp_server_ready", transport="sse", host=sse_host, port=sse_port)
            logger.info(
                "info",
                message="MCP server listening on SSE transport",
                url=f"http://{sse_host}:{sse_port}",
            )
            await server.run_sse(host=sse_host, port=sse_port)

        else:
            logger.error("invalid_transport", transport=transport)
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("mcp_server_shutdown", reason="User interrupt")
    except Exception as e:
        logger.error("mcp_server_fatal_error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
