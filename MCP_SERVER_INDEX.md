# Code Horde MCP Server - Complete Index

## Quick Navigation

### Getting Started (30 minutes)
1. **[MCP_QUICKSTART.md](./MCP_QUICKSTART.md)** - Get running in 5 minutes
2. **[MCP_INTEGRATION_GUIDE.md](./MCP_INTEGRATION_GUIDE.md)** - Step-by-step for Claude Desktop, Claude Code, Cursor

### Reference & Learning (detailed study)
1. **[MCP_SERVER_README.md](./MCP_SERVER_README.md)** - Complete API reference and architecture
2. **[MCP_SERVER_SUMMARY.md](./MCP_SERVER_SUMMARY.md)** - Implementation details and overview
3. **This document** - File index and organization

---

## File Structure

### Core Implementation

#### `src/mcp_server/__init__.py` (10 lines)
**Package initialization and exports**
- Exports `MCPServer` class
- Module docstring

#### `src/mcp_server/server.py` (858 lines)
**Main MCP Server Implementation** ⭐

Key classes and functions:
- `MCPServer` - FastMCP-based server
  - `__init__(api_base_url, api_timeout, redis_url, neo4j_uri)`
  - `_get_http_client()` - Async HTTP client management
  - `_call_api(method, endpoint, **kwargs)` - API communication
  - `_register_tools()` - Define 16 MCP tools
  - `_register_resources()` - Define 4 MCP resources
  - `_register_prompts()` - Define 3 MCP prompt templates
  - `run_stdio()` - Stdio transport runner
  - `run_sse(host, port)` - SSE transport runner

- Response models:
  - `TaskResponse`
  - `AgentStatusResponse`
  - `WorkflowResponse`
  - `SecurityScanResponse`
  - `SystemHealthResponse`

**16 Tools:**
1. `codehorde_create_task` - Create and assign task
2. `codehorde_task_status` - Get task details
3. `codehorde_list_tasks` - List filtered tasks
4. `codehorde_agent_status` - Get all agents status
5. `codehorde_ask_agent` - Query agent by role
6. `codehorde_start_workflow` - Start workflow
7. `codehorde_workflow_status` - Monitor workflow
8. `codehorde_approve` - Approve RED-tier action
9. `codehorde_knowledge_query` - Query knowledge graph
10. `codehorde_knowledge_add` - Add to knowledge graph
11. `codehorde_security_scan` - Trigger security scan
12. `codehorde_security_report` - Get scan results
13. `codehorde_system_health` - Full health check
14. `codehorde_digest` - Activity digest
15. `codehorde_cost_report` - Cost breakdown
16. *(All with comprehensive docstrings for LLM visibility)*

**4 Resources:**
- `codehorde://status` - System status
- `codehorde://agents` - Agent list with capabilities
- `codehorde://policies` - Autonomy policies
- `codehorde://trust-scores` - Trust profiles

**3 Prompts:**
- `delegate_to_army` - Task delegation template
- `security_review` - Security review guide
- `deploy` - Deployment workflow guide

#### `src/mcp_server/client_manager.py` (473 lines)
**External MCP Client Manager** 🔄

Key classes:
- `TransportType` enum - STDIO, SSE, STREAMABLE_HTTP
- `ToolDefinition` - Pydantic model for tool metadata
- `ResourceDefinition` - Pydantic model for resource metadata
- `MCPConnection` - Pydantic model for connection state

- `MCPClientManager` - Main class
  - `connect(server_name, transport, config)` - Connect to external MCP
  - `disconnect(server_name)` - Clean disconnect
  - `list_connections()` - View all connections
  - `get_connection(server_name)` - Get specific connection
  - `list_tools(server_name=None)` - List available tools
  - `discover_tools()` - Full tool definitions
  - `call_tool(server_name, tool_name, arguments)` - Invoke tool
  - `health_check(server_name)` - Monitor connection
  - `reconnect(server_name)` - Automatic reconnection
  - Private methods for tool/resource discovery

**Use Cases:**
- Agents calling GitHub MCP tools
- Agents querying Database MCP
- Agents accessing Filesystem MCP
- Custom enterprise MCP servers

### Configuration

#### `config/mcp_server.yaml` (232 lines)
**Comprehensive Configuration File**

Sections:
- `server` - Name, version, metadata, transports
- `external_servers` - List of external MCPs to connect
- `api` - Code Horde API backend configuration
- `redis` - Optional Redis for caching
- `neo4j` - Knowledge graph connection
- `logging` - Structlog configuration
- `security` - Auth, encryption, rate limiting, CORS
- `tools` - Execution timeouts, caching
- `resources` - Size limits, MIME types
- `prompts` - Caching configuration
- `system` - Health checks, metrics
- `development` - Debug mode, mocking

All values can be overridden by environment variables (documented).

### Startup Script

#### `scripts/start_mcp.py` (172 lines)
**MCP Server Entry Point** 🚀

Features:
- Command-line argument parsing
  - `--transport stdio|sse`
  - `--port PORT`
  - `--host HOST`
  - `--api-url URL`
  - `--help`

- Configuration loading (YAML + env vars)
- Structured logging setup
- Transport selection logic
- Help documentation

Usage:
```bash
python scripts/start_mcp.py [options]
```

### Testing

#### `tests/test_mcp_server.py` (432 lines)
**Comprehensive Unit Tests**

Test classes:
- `TestMCPServer` (6 tests)
  - Initialization
  - HTTP client management
  - API calls (success/failure)
- `TestMCPTools` (1 test)
  - Tool registration
- `TestMCPResources` (1 test)
  - Resource registration
- `TestMCPPrompts` (1 test)
  - Prompt registration
- `TestMCPClientManager` (13 tests)
  - Connection management
  - Tool discovery
  - Tool invocation
  - Error handling
  - Health checks
  - Reconnection logic
- `TestMCPModels` (3 tests)
  - Pydantic model validation

Total: 25+ tests with mocking and async fixtures

Run tests:
```bash
pytest tests/test_mcp_server.py -v
pytest tests/test_mcp_server.py --cov=src.mcp_server
```

---

## Documentation Files

### Quick Start (5-30 minutes)

#### `MCP_QUICKSTART.md` (120 lines)
**Fast path to running the server**
- Prerequisites
- Two startup options (stdio/SSE)
- Integration with Claude Desktop
- Testing methods
- Troubleshooting
- Available tools quick reference

Perfect for: Developers who just want it running NOW.

### Integration & Workflows

#### `MCP_INTEGRATION_GUIDE.md` (591 lines)
**Real-world integration examples**

Sections:
- **Claude Desktop** - Step-by-step config
- **Claude Code** - Auto-discovery setup
- **Cursor** - Configuration and usage
- **External MCP Servers** - GitHub, Filesystem, Database integration
- **Advanced Workflows** (3 scenarios):
  1. Feature development pipeline
  2. Security review and remediation
  3. Multi-tool collaboration with Jira, Slack, GitHub
- **Deployment** - Local development, Docker Compose, production
- **Troubleshooting** - Common issues and fixes
- **Best Practices** - Do's and don'ts

Perfect for: Integrating Code Horde into your workflow.

### Complete Reference

#### `MCP_SERVER_README.md` (556 lines)
**Comprehensive server documentation**

Sections:
- **What is MCP** - Protocol explanation
- **Architecture** - Diagrams and system design
- **Quick Start** - Installation and running
- **Configuration** - All platforms (Claude Desktop, Code, Cursor)
- **Available Tools** (16 tools, full reference)
- **Available Resources** (4 resources, full reference)
- **Available Prompts** (3 prompts, full reference)
- **Code Structure** - Detailed module walkthrough
- **Integration Examples** (4 detailed examples)
- **Dual MCP Integration** - Server and client roles
- **Configuration Reference** - All YAML options
- **Error Handling** - Graceful degradation
- **Security** - Features and considerations
- **Troubleshooting** - Debug guide
- **Performance** - Metrics and optimization
- **Testing** - How to test locally
- **Future Enhancements** - Roadmap

Perfect for: Understanding everything about the MCP server.

### Implementation Details

#### `MCP_SERVER_SUMMARY.md` (453 lines)
**Complete implementation overview**

Sections:
- **What Was Created** - All files with line counts
- **Technical Specifications** - Language, frameworks, architecture
- **File Locations** - Directory structure
- **Key Features** - Tools, resources, prompts
- **Usage Examples** (3 scenarios)
- **Configuration Highlights** - Options and overrides
- **Security Features** - 7 built-in protections
- **Testing** - Coverage and scenarios
- **Performance** - Benchmarks
- **Deployment Options** - Local, Docker, production
- **Integration Points** - With Code Horde and external tools
- **Future Enhancements** - 10 planned features
- **Code Quality Metrics** - Type coverage, documentation, testing
- **Summary** - High-level overview

Perfect for: Understanding what was built and why.

### Navigation (This File)

#### `MCP_SERVER_INDEX.md`
**You are here!** Complete file index and navigation guide.

---

## Code Statistics

| File | Lines | Purpose |
|------|-------|---------|
| `src/mcp_server/__init__.py` | 10 | Package exports |
| `src/mcp_server/server.py` | 858 | Main MCP server |
| `src/mcp_server/client_manager.py` | 473 | External MCP client |
| `config/mcp_server.yaml` | 232 | Configuration |
| `scripts/start_mcp.py` | 172 | Startup script |
| `tests/test_mcp_server.py` | 432 | Unit tests |
| **Subtotal Code** | **2,177** | **Implementation** |
| `MCP_QUICKSTART.md` | 120 | Quick start |
| `MCP_INTEGRATION_GUIDE.md` | 591 | Workflows |
| `MCP_SERVER_README.md` | 556 | Complete reference |
| `MCP_SERVER_SUMMARY.md` | 453 | Implementation details |
| `MCP_SERVER_INDEX.md` | TBD | This file |
| **Subtotal Docs** | **1,720+** | **Documentation** |
| **TOTAL** | **3,900+** | **Complete Project** |

---

## Key Concepts Explained

### What is MCP?

Model Context Protocol - A standardized way for AI models to safely access external tools, resources, and data. Think of it as "how Claude talks to your tools."

### MCP Server (Code Horde's role)

Code Horde EXPOSES itself as an MCP server via `server.py`:
- Claude Desktop/Code/Cursor connects to Code Horde
- Gets access to 16 tools (create tasks, run security scans, etc.)
- Uses prompts and resources for guidance
- Can orchestrate agent workflows from Claude

### MCP Client (Code Horde's other role)

Agents USE external MCP servers via `client_manager.py`:
- Agents connect to GitHub, Filesystem, Database MCPs
- Can create PRs, write files, query databases
- Seamlessly integrate external capabilities
- Makes agents more powerful

### Transports

How communication happens:
- **Stdio**: Direct pipe (best for desktop apps)
- **SSE**: HTTP streaming (best for remote, web)
- **StreamableHTTP**: Alternative HTTP protocol (future)

---

## Common Tasks

### "I just want to run it"
→ Read: [MCP_QUICKSTART.md](./MCP_QUICKSTART.md)

### "I need to use it with Claude Desktop"
→ Read: [MCP_INTEGRATION_GUIDE.md](./MCP_INTEGRATION_GUIDE.md#claude-desktop-integration)

### "I need to use it with Claude Code"
→ Read: [MCP_INTEGRATION_GUIDE.md](./MCP_INTEGRATION_GUIDE.md#claude-code-integration)

### "I need to understand the full API"
→ Read: [MCP_SERVER_README.md](./MCP_SERVER_README.md)

### "I want to see what was built"
→ Read: [MCP_SERVER_SUMMARY.md](./MCP_SERVER_SUMMARY.md)

### "I need advanced workflows"
→ Read: [MCP_INTEGRATION_GUIDE.md#advanced-workflows](./MCP_INTEGRATION_GUIDE.md#advanced-workflows)

### "I need to debug something"
→ Read: [MCP_QUICKSTART.md#troubleshooting](./MCP_QUICKSTART.md#troubleshooting)

### "I want to contribute"
→ Read: [MCP_SERVER_README.md#contributing](./MCP_SERVER_README.md#contributing)

### "I need to deploy to production"
→ Read: [MCP_INTEGRATION_GUIDE.md#deployment-scenarios](./MCP_INTEGRATION_GUIDE.md#deployment-scenarios)

### "I want to understand the code"
→ Read: `server.py` and `client_manager.py` with type hints and docstrings

### "I want to extend it"
→ Read: [MCP_SERVER_README.md#contributing](./MCP_SERVER_README.md#contributing)

---

## Architecture at a Glance

```
┌─ External AI Tools (Claude Desktop, Claude Code, Cursor) ─┐
│  Call 16 Tools + Use 4 Resources + Follow 3 Prompts       │
└─────────────────────────┬──────────────────────────────────┘
                          │ MCP Protocol (stdio/SSE)
                          ▼
            ┌─────────────────────────────┐
            │  Code Horde MCP Server       │
            │  (src/mcp_server/server.py) │
            └─────────┬───────────────────┘
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
   HTTP API              External MCP Servers
   (task management)     (client_manager.py)
   localhost:8000        ├─ GitHub MCP
                         ├─ Filesystem MCP
                         ├─ Database MCP
                         └─ Custom MCPs
```

---

## Implementation Highlights

✅ **16 Production-Ready Tools**
- Task management, agent control, workflows, knowledge graph, security, system monitoring

✅ **4 MCP Resources**
- System status, agent capabilities, policies, trust scores

✅ **3 MCP Prompts**
- Task delegation, security reviews, deployments

✅ **Multiple Transports**
- Stdio (best for desktop), SSE (best for HTTP)

✅ **External MCP Support**
- Agents can use GitHub, Filesystem, Database, and custom MCPs

✅ **Comprehensive Testing**
- 25+ unit tests with mocking and async support

✅ **Production Configuration**
- YAML config with environment overrides, security, logging, monitoring

✅ **Professional Documentation**
- 1,700+ lines across 4 documents with examples

✅ **Type Safety**
- Full type hints, Pydantic models, mypy strict mode

✅ **Error Handling**
- Graceful degradation, detailed logging, no crashes

---

## Next Steps

1. **First Time?** → Start with [MCP_QUICKSTART.md](./MCP_QUICKSTART.md)
2. **Ready to Integrate?** → Follow [MCP_INTEGRATION_GUIDE.md](./MCP_INTEGRATION_GUIDE.md)
3. **Need Details?** → Consult [MCP_SERVER_README.md](./MCP_SERVER_README.md)
4. **Want Code Overview?** → Review [MCP_SERVER_SUMMARY.md](./MCP_SERVER_SUMMARY.md)

---

## Support & Resources

- **Code**: `/sessions/quirky-charming-cori/mnt/code-horde/src/mcp_server/`
- **Config**: `/sessions/quirky-charming-cori/mnt/code-horde/config/mcp_server.yaml`
- **Tests**: `/sessions/quirky-charming-cori/mnt/code-horde/tests/test_mcp_server.py`
- **Startup**: `/sessions/quirky-charming-cori/mnt/code-horde/scripts/start_mcp.py`
- **GitHub**: https://github.com/codehorde/codehorde
- **Issues**: https://github.com/codehorde/codehorde/issues

---

## Document Versions

- **MCP_QUICKSTART.md**: Get running fast
- **MCP_INTEGRATION_GUIDE.md**: Integration workflows
- **MCP_SERVER_README.md**: Complete API reference
- **MCP_SERVER_SUMMARY.md**: Implementation overview
- **MCP_SERVER_INDEX.md**: You are here (navigation hub)

All documents are cross-linked for easy navigation.

---

**Happy coding! 🚀**
