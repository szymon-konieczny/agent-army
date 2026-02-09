# Code Horde MCP Server

## Overview

The Code Horde MCP Server exposes Code Horde's autonomous agent capabilities via the **Model Context Protocol (MCP)**, enabling external AI tools (Claude Code, Claude Desktop, Cursor, etc.) to interact with the agent fleet.

### What is MCP?

The Model Context Protocol is a standardized protocol that allows AI models to safely access external tools, resources, and data sources. It's designed for:
- **Security**: Controlled access with permission models
- **Standardization**: Consistent interface across tools
- **Flexibility**: Supports various transport mechanisms

### Code Horde as MCP Server

Code Horde acts as an **MCP server provider** with two roles:

1. **Server Role** (this module): Exposes Code Horde capabilities to external clients
2. **Client Role** (`client_manager.py`): Code Horde agents can consume external MCP servers

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  External AI Tools (Claude Code, Claude Desktop, Cursor)    │
└───────────────────────┬─────────────────────────────────────┘
                        │ MCP Protocol
                        │ (stdio/SSE)
                        ▼
        ┌───────────────────────────────┐
        │   Code Horde MCP Server        │
        │  (src/mcp_server/server.py)   │
        └───────────────┬───────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
   HTTP API              External MCP Servers
   (Code Horde)           (client_manager.py)
   localhost:8000           ├── GitHub MCP
                            ├── Filesystem MCP
                            ├── Database MCP
                            └── Custom tools
```

## Quick Start

### Installation

The MCP server requires FastMCP, which needs to be added to dependencies. For now, ensure your environment has:

```bash
pip install anthropic
# FastMCP will be available in upcoming releases
```

### Running the Server

#### Method 1: Stdio Transport (Recommended for Claude Desktop/Code)

```bash
python scripts/start_mcp.py --transport stdio
```

This creates a bidirectional pipe for communication with Claude Desktop or Claude Code.

#### Method 2: SSE Transport (HTTP)

```bash
python scripts/start_mcp.py --transport sse --port 8001
```

Creates an HTTP server with Server-Sent Events streaming.

### Configuration

#### For Claude Desktop

Add to `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "codehorde": {
      "command": "python",
      "args": [
        "/path/to/code-horde/scripts/start_mcp.py",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

#### For Claude Code

Claude Code auto-detects and connects to MCP servers running on stdio. Simply run:

```bash
python scripts/start_mcp.py
```

And Claude Code will discover it automatically.

#### Environment Variables

Override configuration via environment variables:

```bash
# API endpoint
AGENTARMY_MCP_API_URL=http://api.example.com:8000

# Transport
AGENTARMY_MCP_TRANSPORT=stdio  # or sse

# SSE settings
AGENTARMY_MCP_SSE_HOST=0.0.0.0
AGENTARMY_MCP_SSE_PORT=8001

# Database credentials
AGENTARMY_NEO4J_PASSWORD=your_password
AGENTARMY_REDIS_URL=redis://localhost:6379/0
```

## Available Tools

The MCP server exposes 16 tools that external AI can use:

### Task Management

- **`codehorde_create_task`** - Create a task and auto-assign to best agent
  ```
  Input: description (str), priority (1-5), tags (list)
  Output: task_id, status, assigned_agent
  ```

- **`codehorde_task_status`** - Get status of a specific task
  ```
  Input: task_id (str)
  Output: task details, progress, result
  ```

- **`codehorde_list_tasks`** - List tasks with filtering
  ```
  Input: status (all/pending/in_progress/completed/failed), limit (int)
  Output: tasks array, total count
  ```

### Agent Operations

- **`codehorde_agent_status`** - Get status of all agents
  ```
  Output: agents array with role, state, trust_score, capabilities
  ```

- **`codehorde_ask_agent`** - Query a specific agent type
  ```
  Input: agent_role (sentinel/executor/guardian), question (str)
  Output: agent response and reasoning
  ```

### Workflow Execution

- **`codehorde_start_workflow`** - Start a predefined workflow
  ```
  Input: workflow_name (feature_development/security_scan/bug_fix/deploy_*), context (dict)
  Output: execution_id, status, current_step
  ```

- **`codehorde_workflow_status`** - Check workflow progress
  ```
  Input: execution_id (str)
  Output: status, progress (0-100), current_step
  ```

- **`codehorde_approve`** - Approve RED-tier actions
  ```
  Input: request_id (str)
  Output: approval confirmation, action status
  ```

### Knowledge Graph

- **`codehorde_knowledge_query`** - Query Neo4j knowledge graph
  ```
  Input: question (str)
  Output: entities, relationships, answers
  ```

- **`codehorde_knowledge_add`** - Add knowledge to graph
  ```
  Input: entity (str), entity_type (str), relationships (list)
  Output: entity_id, relationship_count
  ```

### Security

- **`codehorde_security_scan`** - Trigger security scan
  ```
  Input: target (all/code/dependencies/secrets/config)
  Output: scan_id, status, findings
  ```

- **`codehorde_security_report`** - Get latest security report
  ```
  Output: findings, severity_summary, scan_timestamp
  ```

### System Operations

- **`codehorde_system_health`** - Full system health check
  ```
  Output: healthy (bool), agents_online, pending_tasks, trust_scores, uptime
  ```

- **`codehorde_digest`** - Get activity digest
  ```
  Output: recent_events, statistics, completed_tasks
  ```

- **`codehorde_cost_report`** - LLM cost breakdown
  ```
  Input: period (today/week/month/all)
  Output: cost_summary, per_agent, per_model, projections
  ```

## Available Resources

Resources provide static context and configuration:

### `codehorde://status`
Current system status (agents online, task queue depth, health status).

### `codehorde://agents`
List of all agents with capabilities, roles, and trust scores.

### `codehorde://policies`
Current autonomy policies (BLUE/GREEN/RED levels).

### `codehorde://trust-scores`
Trust profiles for all agents.

## Available Prompts

Prompt templates guide AI in using the tools effectively:

### `delegate_to_army(task_description)`
Template for delegating work to Code Horde with proper structure and safety considerations.

### `security_review(target)`
Template for security reviews using agents (code, dependencies, config).

### `deploy(environment)`
Template for safe deployments with approval workflows.

## Code Structure

```
src/mcp_server/
├── __init__.py           # Package exports
├── server.py             # Main MCP server (500+ lines)
│   ├── MCPServer class
│   ├── Tool definitions
│   ├── Resource definitions
│   ├── Prompt templates
│   ├── run_stdio()
│   └── run_sse()
└── client_manager.py     # External MCP client (300+ lines)
    ├── MCPClientManager class
    ├── MCPConnection model
    ├── connect()
    ├── list_tools()
    └── call_tool()

scripts/
└── start_mcp.py          # Startup script

config/
└── mcp_server.yaml       # Configuration file
```

### server.py Overview

The main server implementation (~500 lines) includes:

**Core Components:**
- `MCPServer` class - Main server with FastMCP integration
- Tool implementations (16 tools, ~30-50 lines each)
- Resource definitions (4 resources)
- Prompt templates (3 prompts)
- Async HTTP client for API communication

**Transport Support:**
- Stdio (stdout/stdin bidirectional)
- SSE (Server-Sent Events over HTTP)

**Error Handling:**
- Comprehensive exception handling
- Graceful degradation when API unavailable
- Detailed logging with structlog

### client_manager.py Overview

MCP client for consuming external servers (~300 lines):

**Key Classes:**
- `MCPConnection` - Connection state and metadata
- `MCPClientManager` - Connection and tool management

**Key Methods:**
- `connect()` - Establish connection to external MCP server
- `list_tools()` - Discover available tools
- `call_tool()` - Invoke tools on external servers
- `health_check()` - Monitor connection health
- `reconnect()` - Automatic reconnection logic

## Integration Examples

### Example 1: Create a Task from Claude Code

```python
# Claude Code can now call:
result = await mcp.call("codehorde_create_task", {
    "description": "Implement user authentication feature",
    "priority": 2,
    "tags": ["feature", "security"]
})

# Returns:
# {
#   "task_id": "task-12345",
#   "status": "assigned",
#   "assigned_agent": "executor-1",
#   "created_at": "2024-02-06T15:30:00Z"
# }
```

### Example 2: Monitor Deployment with Prompts

```python
# Claude Code uses the deployment prompt
prompt = mcp.get_prompt("deploy", {"environment": "production"})

# Prompt guides Claude to:
# 1. Run security scans
# 2. Get approval workflow
# 3. Use codehorde_start_workflow
# 4. Monitor with codehorde_workflow_status
```

### Example 3: Query Knowledge Graph

```python
# Ask about system architecture
result = await mcp.call("codehorde_knowledge_query", {
    "question": "What services depend on the auth microservice?"
})

# Returns graph relationships and insights
```

### Example 4: Code Horde Using External MCP Tools

```python
# From agent code, use client_manager:
from src.mcp_server.client_manager import MCPClientManager

client_mgr = MCPClientManager()

# Connect to GitHub MCP
await client_mgr.connect("github", TransportType.SSE, {
    "url": "http://localhost:3000"
})

# Use GitHub tools
result = await client_mgr.call_tool("github", "create_issue", {
    "repo": "myorg/myrepo",
    "title": "Bug fix deployed",
    "body": "..."
})
```

## Dual MCP Integration

Code Horde uniquely operates as both MCP server and client:

```
Claude Code
    │
    ├─→ [MCP Server]
    │   ├─→ create_task
    │   ├─→ security_scan
    │   └─→ start_workflow
    │
    ← Creates tasks →
    │
    ▼
Code Horde Agents
    │
    ├─→ [MCP Client Manager]
    │   ├─→ GitHub MCP (create PRs)
    │   ├─→ Filesystem MCP (file access)
    │   └─→ Database MCP (queries)
    │
    └─→ Creates/calls external tools
```

## Configuration

See `config/mcp_server.yaml` for comprehensive options:

- **Server identity** - Name, version, metadata
- **Transport settings** - Stdio and SSE configuration
- **External servers** - MCP servers to consume
- **API configuration** - Code Horde backend connection
- **Database** - Redis and Neo4j settings
- **Logging** - Structlog configuration
- **Security** - Auth, encryption, rate limiting
- **Tool/Resource config** - Timeouts, caching, size limits

## Error Handling

All tools include comprehensive error handling:

```python
try:
    result = await self._call_api("POST", "/api/tasks", json=payload)
except httpx.HTTPError as e:
    logger.error("api_call_failed", endpoint=endpoint, error=str(e))
    return {"error": str(e), "status": "failed"}
```

**Graceful Degradation:**
- Returns error details instead of raising
- Logs all failures with context
- MCP client can decide action
- No crashes or hangs

## Security Considerations

1. **API Authentication** - Configure in yaml (future: bearer tokens)
2. **TLS/SSL** - Supported for SSE transport
3. **Rate Limiting** - Configurable per endpoint
4. **CORS** - Restricted to localhost and Claude apps
5. **External MCP Connections** - Validate before connecting
6. **Tool Arguments** - All inputs validated via Pydantic

## Troubleshooting

### Server won't start

```bash
# Check Python version
python --version  # Must be 3.12+

# Check dependencies
pip install pydantic structlog httpx fastapi uvicorn

# Enable verbose logging
AGENTARMY_MCP_DEBUG=1 python scripts/start_mcp.py
```

### API connection failed

```bash
# Check Code Horde API is running
curl http://localhost:8000/api/health

# Override API URL
AGENTARMY_MCP_API_URL=http://api.example.com:8000 python scripts/start_mcp.py
```

### Claude Code doesn't detect server

```bash
# Ensure stdio transport is used
python scripts/start_mcp.py --transport stdio

# Check logs
python scripts/start_mcp.py 2>&1 | grep -i "ready\|listening"
```

### Tools return empty or errors

```bash
# Check API connectivity
python scripts/start_mcp.py --api-url http://localhost:8000

# Check logs for API errors
# All API calls are logged with full context
```

## Performance

- **Tool calls:** <100ms average (local API)
- **Concurrent calls:** Limited to ~10 (configurable)
- **Resource caching:** 300s TTL (configurable)
- **Memory footprint:** ~50MB (single HTTP client)

## Testing

Test the MCP server locally:

```bash
# Terminal 1: Start Code Horde API
python -m src.api.main

# Terminal 2: Start MCP server
python scripts/start_mcp.py --transport stdio

# Terminal 3: Test with Claude Code or curl
python -c "
import asyncio
from src.mcp_server.server import MCPServer

async def test():
    server = MCPServer()
    # Tools available via server.app
    print('MCP Server initialized')

asyncio.run(test())
"
```

## Future Enhancements

1. **WebSocket Transport** - Real-time bidirectional communication
2. **Tool Caching** - Cache tool results for repeated calls
3. **Streaming Results** - Large result streaming
4. **Batch Operations** - Multiple tools in single call
5. **Custom Tools** - Register agent-specific tools
6. **Analytics** - Track tool usage and performance
7. **Multi-tenancy** - Support multiple organizations
8. **GraphQL API** - Alternative query interface

## Contributing

To add new tools:

1. Create async function in `server.py`
2. Decorate with `@self.app.tool()`
3. Add comprehensive docstring (shown to LLM)
4. Handle errors gracefully
5. Log operations with structlog
6. Update this README

## License

MIT - See LICENSE file

## Support

- GitHub Issues: https://github.com/codehorde/codehorde/issues
- Documentation: https://codehorde.dev
- Email: team@codehorde.dev
