# AgentArmy MCP Integration Guide

This guide shows how to integrate AgentArmy's MCP server with various AI tools and external MCP servers.

## Table of Contents

1. [Claude Desktop Integration](#claude-desktop-integration)
2. [Claude Code Integration](#claude-code-integration)
3. [Cursor Integration](#cursor-integration)
4. [External MCP Server Integration](#external-mcp-server-integration)
5. [Advanced Workflows](#advanced-workflows)

---

## Claude Desktop Integration

### Step 1: Get Claude Desktop

Download from [claude.ai](https://claude.ai/download) or your platform's app store.

### Step 2: Find Configuration File

The configuration file location depends on your OS:

**macOS:**
```
~/.config/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```
~/.config/Claude/claude_desktop_config.json
```

### Step 3: Add MCP Server Configuration

If the file doesn't exist, create it. Add the AgentArmy MCP server:

```json
{
  "mcpServers": {
    "agentarmy": {
      "command": "python",
      "args": [
        "/path/to/agent-army/scripts/start_mcp.py",
        "--transport",
        "stdio"
      ],
      "env": {
        "AGENTARMY_MCP_API_URL": "http://localhost:8000",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

**Replace `/path/to/agent-army` with your actual path.**

### Step 4: Verify Installation

1. Restart Claude Desktop
2. Open a conversation
3. Click the settings gear icon
4. Look for "Model Context Protocol" section
5. You should see "agentarmy" listed under connected servers

### Step 5: Use AgentArmy Tools

In any conversation, you can now:

```
I need you to help me deploy a feature. Can you:
1. Create a task for the feature implementation
2. Run a security scan
3. Monitor the deployment workflow

Use the agentarmy tools available to you.
```

Claude Desktop will:
- See available tools from AgentArmy MCP
- Call appropriate tools during conversation
- Display results in the chat
- Use prompts to guide workflow

---

## Claude Code Integration

Claude Code (GitHub Copilot integration) auto-discovers MCP servers. It's easier than Desktop!

### Step 1: Start the MCP Server

```bash
cd /path/to/agent-army
python scripts/start_mcp.py --transport stdio
```

Keep this running in a terminal.

### Step 2: No Additional Configuration Needed

Claude Code automatically discovers MCP servers running on stdio transport.

### Step 3: Use in VS Code

Open any file in VS Code with Claude Code, and type:

```
# Create a task to implement feature X
# Use the agentarmy tools to create and monitor it
```

Claude Code will:
- Detect available tools
- Call agentarmy_create_task
- Monitor progress with agentarmy_task_status
- Integrate results into code generation

### Step 4: Working with Agents from VS Code

```python
# example.py
def deploy_feature():
    # Create task
    # Use agentarmy MCP tools to:
    # 1. agentarmy_create_task - create deployment task
    # 2. agentarmy_start_workflow - start deployment workflow
    # 3. agentarmy_workflow_status - monitor progress
    pass
```

---

## Cursor Integration

Cursor has excellent MCP support through its configuration.

### Step 1: Locate Cursor Config

**macOS/Linux:**
```
~/.cursor/config.json
```

**Windows:**
```
%APPDATA%\.cursor\config.json
```

### Step 2: Add MCP Server

```json
{
  "mcpServers": {
    "agentarmy": {
      "type": "stdio",
      "command": "python",
      "args": [
        "/path/to/agent-army/scripts/start_mcp.py",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

### Step 3: Verify in Cursor

- Restart Cursor
- Open a file
- Press `Cmd+K` (macOS) or `Ctrl+K` (Windows/Linux)
- Ask Cursor to use AgentArmy tools
- Cursor will automatically list available MCP tools

### Step 4: Example Usage in Cursor

```
@agentarmy Create a security scanning task and report findings
```

Cursor will automatically:
- Call `agentarmy_security_scan`
- Get results with `agentarmy_security_report`
- Format findings in your editor

---

## External MCP Server Integration

AgentArmy agents can consume external MCP servers (GitHub, Filesystem, Database, etc.).

### Step 1: Connect External MCP Server

From agent code, use the MCPClientManager:

```python
from src.mcp_server.client_manager import MCPClientManager, TransportType

# Initialize manager
client_mgr = MCPClientManager()

# Connect to GitHub MCP
await client_mgr.connect(
    "github",
    TransportType.SSE,
    {"url": "http://localhost:3000"}
)

# List available tools
tools = client_mgr.list_tools("github")
# ['create_issue', 'create_pr', 'search_issues', ...]

# Call a tool
result = await client_mgr.call_tool("github", "create_issue", {
    "repo": "myorg/myrepo",
    "title": "Bug: authentication failing",
    "body": "Users report unable to login on mobile devices",
})
```

### Step 2: Configure External Servers in YAML

Edit `config/mcp_server.yaml`:

```yaml
external_servers:
  - name: "github"
    transport: "sse"
    config:
      url: "http://localhost:3000"

  - name: "filesystem"
    transport: "stdio"
    config:
      command: "python"
      args: ["-m", "mcp.servers.filesystem", "/workspace"]

  - name: "database"
    transport: "streamable-http"
    config:
      url: "http://localhost:8002"
```

### Step 3: Use in Agent Workflows

Agents automatically gain access to connected MCP servers:

```python
class FeatureExecutor(BaseAgent):
    """Executes feature development tasks."""

    async def execute_task(self, task: Task) -> TaskResult:
        # Create GitHub issue from task
        issue = await self.mcp_client.call_tool("github", "create_issue", {
            "repo": self.config.target_repo,
            "title": task.description,
        })

        # Create files with filesystem MCP
        await self.mcp_client.call_tool("filesystem", "create_file", {
            "path": f"src/features/{task.id}.py",
            "content": "# New feature"
        })

        # Query database for dependencies
        deps = await self.mcp_client.call_tool("database", "query", {
            "sql": "SELECT * FROM modules WHERE feature_id = ?",
            "params": [task.id]
        })

        return TaskResult(...)
```

---

## Advanced Workflows

### Workflow 1: Full Feature Development Pipeline

```
Claude Desktop User
    │
    ├─ "Create feature X and deploy to staging"
    │
    ▼
AgentArmy MCP Server (Claude's view)
    ├─ agentarmy_create_task(description="Implement feature X")
    ├─ agentarmy_security_scan(target="all")
    ├─ agentarmy_start_workflow(workflow_name="feature_development")
    ├─ agentarmy_workflow_status(execution_id)
    │
    ▼
AgentArmy Agents (internal execution)
    ├─ Executor creates GitHub PR via GitHub MCP
    ├─ CI/CD runs (external trigger)
    ├─ Sentinel runs security scan
    ├─ Guardian collects approvals
    │
    ▼
External MCP Servers
    ├─ GitHub MCP: PR operations, reviews
    ├─ Slack MCP: Notification to team
    ├─ Database MCP: Log execution
    │
    └─ Results flow back to Claude
```

### Workflow 2: Security Review and Remediation

**User Request (Claude Desktop):**
```
"Run a security scan on our codebase and fix any vulnerabilities"
```

**Claude's Action:**
```
1. agentarmy_security_scan(target="all")
   → Returns: scan_id, findings

2. agentarmy_knowledge_query(
     question="What are common remediation patterns?"
   )
   → Returns: Fix strategies from knowledge graph

3. agentarmy_ask_agent(
     agent_role="sentinel",
     question="What's the risk level of these findings?"
   )
   → Returns: Risk assessment

4. agentarmy_create_task(
     description="Fix critical vulnerabilities found in scan",
     priority=1,
     tags=["security", "critical"]
   )
   → Returns: task_id, assigned agent

5. Monitor with agentarmy_task_status(task_id)
```

**Agent Execution:**
```python
# Sentinel agent gets assigned
async def execute_security_fixes(task: Task):
    # Get findings from knowledge graph
    findings = await self.knowledge.query(
        "Get findings from scan " + task.payload["scan_id"]
    )

    # For each finding, create fix
    for finding in findings:
        # Create branch with filesystem MCP
        branch = await mcp_client.call_tool("github", "create_branch", {
            "branch": f"security/fix-{finding.id}",
            "from": "main"
        })

        # Apply fix using filesystem MCP
        fixed_code = generate_fix(finding)
        await mcp_client.call_tool("filesystem", "write_file", {
            "path": finding.file_path,
            "content": fixed_code
        })

        # Create PR for review
        pr = await mcp_client.call_tool("github", "create_pr", {
            "title": f"Security: Fix {finding.title}",
            "branch": branch,
            "description": f"Fixes {finding.severity} vulnerability"
        })

    return TaskResult(status="completed", output={
        "prs_created": len(findings),
        "severity_fixed": "critical"
    })
```

### Workflow 3: Multi-Tool Collaboration

**Setup:**
```yaml
external_servers:
  - name: "github"     # Code management
  - name: "jira"       # Issue tracking
  - name: "slack"      # Team communication
  - name: "database"   # Data queries
```

**Workflow in Agent Code:**
```python
async def dev_task_flow():
    # 1. Create issue in Jira
    jira_issue = await mcp_mgr.call_tool("jira", "create_issue", {
        "project": "DEV",
        "issue_type": "Story",
        "summary": task.description
    })

    # 2. Notify team on Slack
    await mcp_mgr.call_tool("slack", "send_message", {
        "channel": "#development",
        "text": f"Starting task: {jira_issue.key}"
    })

    # 3. Create GitHub branch
    branch = await mcp_mgr.call_tool("github", "create_branch", {
        "from": "main",
        "name": f"feature/{jira_issue.key.lower()}"
    })

    # 4. Query database for context
    context = await mcp_mgr.call_tool("database", "query", {
        "sql": "SELECT * FROM relevant_data WHERE project_id = ?",
        "params": [task.project_id]
    })

    # 5. Log progress
    await mcp_mgr.call_tool("database", "write", {
        "table": "execution_log",
        "data": {"task_id": task.id, "status": "started", ...}
    })

    # 6. Execute actual work...

    # 7. Create GitHub PR when done
    pr = await mcp_mgr.call_tool("github", "create_pr", {
        "branch": branch,
        "title": f"[{jira_issue.key}] {task.description}"
    })

    # 8. Update Jira
    await mcp_mgr.call_tool("jira", "update_issue", {
        "key": jira_issue.key,
        "status": "In Progress",
        "github_pr": pr.url
    })

    # 9. Final team update
    await mcp_mgr.call_tool("slack", "send_message", {
        "channel": "#development",
        "text": f"PR ready for review: {pr.url}"
    })
```

---

## Deployment Scenarios

### Local Development

```bash
# Terminal 1: Start AgentArmy API
python -m src.api.main --host 0.0.0.0 --port 8000

# Terminal 2: Start MCP Server
python scripts/start_mcp.py --transport stdio

# Terminal 3: Claude Desktop connects automatically
# Or Claude Code detects stdio server
```

### Production Deployment

**Docker Compose:**
```yaml
version: '3'
services:
  agentarmy-api:
    image: agentarmy:latest
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - NEO4J_URI=bolt://neo4j:7687

  agentarmy-mcp:
    image: agentarmy-mcp:latest
    ports:
      - "8001:8001"
    environment:
      - AGENTARMY_MCP_API_URL=http://agentarmy-api:8000
      - AGENTARMY_MCP_TRANSPORT=sse
      - AGENTARMY_MCP_SSE_PORT=8001
    depends_on:
      - agentarmy-api

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  neo4j:
    image: neo4j:5
    ports:
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
```

Then configure clients to connect to `http://mcp-server:8001` (SSE).

---

## Troubleshooting

### MCP Server Not Detected

**Claude Desktop:**
```bash
# Check config file
cat ~/.config/Claude/claude_desktop_config.json

# Verify path exists
ls /path/to/agent-army/scripts/start_mcp.py

# Try running directly
python /path/to/agent-army/scripts/start_mcp.py --transport stdio
```

**Claude Code:**
```bash
# Ensure stdio transport
python scripts/start_mcp.py --transport stdio

# Check for output
python scripts/start_mcp.py 2>&1 | grep -i ready
```

### Tools Return Empty Results

```bash
# Check AgentArmy API is running
curl http://localhost:8000/api/health

# Check logs
python scripts/start_mcp.py 2>&1 | tail -20

# Enable verbose logging
AGENTARMY_MCP_DEBUG=1 python scripts/start_mcp.py
```

### External MCP Connection Failed

```python
# Test connection
from src.mcp_server.client_manager import MCPClientManager, TransportType

mgr = MCPClientManager()
try:
    await mgr.connect("test", TransportType.SSE, {"url": "http://localhost:3000"})
    print("Connected!")
except Exception as e:
    print(f"Connection failed: {e}")
```

---

## Best Practices

1. **Keep MCP Server Running** - Use process manager (systemd, supervisord)
2. **Version Control Config** - Track configuration in git
3. **Use Environment Variables** - Don't hardcode sensitive data
4. **Monitor Logs** - Set up log aggregation
5. **Test Tool Chains** - Verify external MCP connections
6. **Rate Limit** - Configure tool execution limits
7. **Cache Results** - Enable result caching in config
8. **Secure Communication** - Use TLS in production

---

## Next Steps

- [Read MCP_SERVER_README.md](./MCP_SERVER_README.md) for full API reference
- [Check test_mcp_server.py](./tests/test_mcp_server.py) for examples
- [Explore agent implementation](./src/core/agent_base.py) to see MCP usage

---

## Support

- Issues: https://github.com/agentarmy/agentarmy/issues
- Docs: https://agentarmy.dev
- Community: Discord (link in GitHub)
