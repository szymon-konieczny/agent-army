# Code Horde — Secure Multi-Agent System Architecture

## 1. Vision

Code Horde is a secure, self-hosted multi-agent AI system designed to autonomously develop, monitor, and protect SaaS applications — starting with **Prooflog** (blockchain-based tamper-proof logging). The system communicates with its operator via **WhatsApp** and is architected for future migration to a cloud-based Command Center.

### Design Principles

- **Zero-Trust Security** — every agent interaction is authenticated, authorized, and audited
- **Defense in Depth** — multiple security layers (container isolation, capability-based permissions, encrypted channels)
- **Immutable Audit Trail** — all agent actions logged to tamper-proof store (inspired by Prooflog's own architecture)
- **Multi-Model Intelligence** — Claude API for complex reasoning, local Ollama models for sensitive data
- **Protocol-First** — A2A (Agent-to-Agent) + MCP (Model Context Protocol) compliance
- **Cloud-Ready** — abstractions that enable seamless migration from local to cloud deployment

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HUMAN INTERFACE LAYER                        │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│   │   WhatsApp    │    │  Future: Web  │    │  Future: Cloud CLI  │ │
│   │   Bridge      │    │  Dashboard    │    │  Command Center     │ │
│   └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘ │
│          │                   │                        │             │
└──────────┼───────────────────┼────────────────────────┼─────────────┘
           │                   │                        │
           ▼                   ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        GATEWAY / API LAYER                          │
│                                                                     │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │                    FastAPI Gateway                           │   │
│   │  • Authentication (JWT + API Keys)                          │   │
│   │  • Rate Limiting                                            │   │
│   │  • Request Validation (Pydantic)                            │   │
│   │  • Webhook Handlers (WhatsApp, GitHub, etc.)                │   │
│   └────────────────────────┬───────────────────────────────────┘   │
│                            │                                        │
└────────────────────────────┼────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                              │
│                                                                     │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │                    Commander Agent                          │   │
│   │  • Task Routing & Decomposition                             │   │
│   │  • Workflow State Machine                                   │   │
│   │  • Priority Queue Management                                │   │
│   │  • Agent Health Monitoring                                  │   │
│   │  • Human-in-the-Loop Escalation                             │   │
│   └────────────┬──────────────────────────┬────────────────────┘   │
│                │                          │                         │
│   ┌────────────▼──────────┐  ┌───────────▼───────────────┐        │
│   │  Redis Message Bus    │  │  RabbitMQ Task Queues      │        │
│   │  (fast, ephemeral)    │  │  (durable, complex routing)│        │
│   └───────────────────────┘  └───────────────────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AGENT LAYER                                  │
│                                                                     │
│   ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────────┐ │
│   │ Sentinel  │ │  Builder  │ │ Inspector │ │     Watcher       │ │
│   │ (Security)│ │  (Dev)    │ │ (QA)      │ │   (Monitoring)    │ │
│   └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └────────┬──────────┘ │
│         │             │             │                  │            │
│   ┌─────┴─────┐ ┌─────┴─────┐ ┌─────┴─────┐ ┌────────┴──────────┐ │
│   │  Scout    │ │  Scribe   │ │  DevOps   │ │   Analyst         │ │
│   │ (Research)│ │  (Docs)   │ │ (Infra)   │ │   (Data)          │ │
│   └───────────┘ └───────────┘ └───────────┘ └───────────────────┘ │
│                                                                     │
│   Each agent runs in isolated Docker container with:                │
│   • Unique identity (X.509 cert or JWT)                             │
│   • Capability-based permissions                                    │
│   • Resource limits (CPU, memory, network)                          │
│   • Dedicated audit log stream                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     INTELLIGENCE LAYER                               │
│                                                                     │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │                Multi-Model Router                           │   │
│   │                                                             │   │
│   │  ┌─────────────────┐     ┌──────────────────────┐          │   │
│   │  │  Claude API      │     │  Ollama Local         │          │   │
│   │  │  (Opus/Sonnet)   │     │  (Qwen, Llama, etc.) │          │   │
│   │  │                  │     │                       │          │   │
│   │  │  Complex tasks:  │     │  Sensitive data:      │          │   │
│   │  │  • Architecture  │     │  • Secret scanning    │          │   │
│   │  │  • Code review   │     │  • Local analysis     │          │   │
│   │  │  • Security audit│     │  • PII processing     │          │   │
│   │  │  • Planning      │     │  • Embeddings         │          │   │
│   │  └─────────────────┘     └──────────────────────┘          │   │
│   │                                                             │   │
│   │  Routing Rules:                                             │   │
│   │  • Sensitivity level → local vs cloud                       │   │
│   │  • Complexity score → model tier                            │   │
│   │  • Cost budget → optimization                               │   │
│   │  • Fallback chain → Claude → Ollama → queue for human       │   │
│   └────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SECURITY LAYER                                 │
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│   │  AuthN/AuthZ │  │  Secrets Mgr │  │  Immutable Audit Log     │ │
│   │  (JWT/mTLS)  │  │  (Vault)     │  │  (Hash-chained entries)  │ │
│   └──────────────┘  └──────────────┘  └──────────────────────────┘ │
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│   │  Sandbox     │  │  Policy      │  │  Threat Detection        │ │
│   │  (Docker)    │  │  Engine      │  │  (Prompt Injection, etc.)│ │
│   └──────────────┘  └──────────────┘  └──────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PERSISTENCE LAYER                              │
│                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│   │  PostgreSQL  │  │  Redis       │  │  File Storage            │ │
│   │  (State,     │  │  (Cache,     │  │  (Artifacts,             │ │
│   │   Audit,     │  │   Sessions,  │  │   Generated Code,        │ │
│   │   History)   │  │   Pub/Sub)   │  │   Reports)               │ │
│   └──────────────┘  └──────────────┘  └──────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Agent Roster

### Commander (Orchestrator)
- **Role**: Central coordinator, task router, workflow manager
- **Capabilities**: Task decomposition, agent selection, priority management, escalation
- **Security Level**: CRITICAL — access to all agent APIs
- **LLM**: Claude Opus (complex planning) / Sonnet (routine routing)

### Sentinel (Security Agent)
- **Role**: Security monitoring, vulnerability scanning, threat detection
- **Capabilities**: Code scanning, dependency audit, secret detection, penetration testing coordination
- **Security Level**: CRITICAL — read-only access to all systems, write to security logs
- **LLM**: Claude (security analysis) + Local (secret scanning)

### Builder (Development Agent)
- **Role**: Code generation, PR creation, bug fixes, feature implementation
- **Capabilities**: Git operations, code writing, test generation, refactoring
- **Security Level**: HIGH — write access to repos, no production access
- **LLM**: Claude Opus (architecture) / Sonnet (implementation)

### Inspector (QA Agent)
- **Role**: Testing, validation, quality assurance
- **Capabilities**: Test execution, coverage analysis, regression detection, performance testing
- **Security Level**: MEDIUM — read repos, write test results
- **LLM**: Claude Sonnet (test generation) / Local (test execution analysis)

### Watcher (Monitoring Agent)
- **Role**: System health, performance monitoring, alerting
- **Capabilities**: Metrics collection, anomaly detection, uptime monitoring, log analysis
- **Security Level**: MEDIUM — read-only system metrics, write alerts
- **LLM**: Local (routine monitoring) / Claude (anomaly analysis)

### Scout (Research Agent)
- **Role**: Technology research, competitive analysis, threat intelligence
- **Capabilities**: Web search, documentation analysis, trend monitoring
- **Security Level**: LOW — internet access only, no internal system access
- **LLM**: Claude Sonnet (research synthesis)

### Scribe (Documentation Agent)
- **Role**: Documentation generation, knowledge base maintenance
- **Capabilities**: Doc generation, changelog creation, API documentation
- **Security Level**: MEDIUM — read code, write docs
- **LLM**: Claude Sonnet (writing) / Local (formatting)

### DevOps (Infrastructure Agent)
- **Role**: Deployment, infrastructure management, CI/CD
- **Capabilities**: Docker management, deployment orchestration, configuration management
- **Security Level**: HIGH — infrastructure access with approval gates
- **LLM**: Claude (planning) / Local (config generation)

---

## 4. Security Architecture

### 4.1 Zero-Trust Model

```
Every agent interaction follows:

1. IDENTIFY  → Agent presents credentials (JWT/cert)
2. VERIFY    → Orchestrator validates identity + capabilities
3. AUTHORIZE → Policy engine checks if action is permitted
4. EXECUTE   → Action runs in sandboxed environment
5. AUDIT     → Result logged to immutable audit trail
6. REVIEW    → Anomaly detection on audit stream
```

### 4.2 Capability-Based Permissions

Each agent has a capability manifest (YAML):

```yaml
# Example: Builder agent capabilities
agent_id: builder-001
capabilities:
  git:
    - read_repository
    - create_branch
    - push_commits
    - create_pull_request
  filesystem:
    - read: ["/workspace/src/**", "/workspace/tests/**"]
    - write: ["/workspace/src/**", "/workspace/tests/**"]
    - deny: ["/workspace/.env", "/workspace/secrets/**"]
  network:
    - allow: ["github.com", "pypi.org", "npmjs.org"]
    - deny: ["*"]  # default deny
  llm:
    - claude_sonnet
    - claude_opus  # requires escalation approval
    - ollama_local
  dangerous_operations:
    requires_approval:
      - delete_branch
      - force_push
      - modify_ci_config
      - deploy_to_staging
```

### 4.3 Immutable Audit Trail

Inspired by Prooflog's blockchain logging:

```
┌──────────────────────────────────────────────────┐
│  Audit Entry #N                                   │
│                                                    │
│  timestamp:    2026-02-06T20:45:00Z               │
│  agent_id:     builder-001                         │
│  action:       git.push_commits                    │
│  target:       prooflog/backend:feature/auth       │
│  input_hash:   sha256:a1b2c3...                    │
│  output_hash:  sha256:d4e5f6...                    │
│  prev_hash:    sha256:789abc...  ◄── chain link    │
│  entry_hash:   sha256:def012...  ◄── self hash     │
│  signature:    ed25519:...       ◄── agent sig     │
│                                                    │
└──────────────────────────────────────────────────┘
```

Each entry is hash-chained to the previous one, making tampering detectable.

### 4.4 Threat Model

| Threat | Mitigation |
|---|---|
| Prompt injection via external content | Input sanitization + content isolation + LLM output validation |
| Compromised agent lateral movement | Container isolation + network segmentation + capability limits |
| Secret exfiltration | Secrets never in agent memory; accessed via Vault API per-request |
| Unauthorized escalation | Multi-party approval for critical operations |
| Supply chain attack | Dependency pinning + hash verification + isolated build environment |
| Audit log tampering | Hash-chained entries + periodic integrity verification |

---

## 5. Communication Architecture

### 5.1 WhatsApp Bridge

```
User (WhatsApp) ──► WhatsApp Cloud API ──► Webhook (FastAPI)
                                                    │
                                                    ▼
                                            Message Parser
                                                    │
                                            ┌───────┴────────┐
                                            │                │
                                      Command?          Conversation?
                                            │                │
                                            ▼                ▼
                                      Task Queue      Context Manager
                                            │                │
                                            ▼                ▼
                                      Commander         LLM Router
                                            │                │
                                            ▼                ▼
                                      Agent Execution   Response Gen
                                            │                │
                                            └───────┬────────┘
                                                    │
                                                    ▼
                                            Response Formatter
                                                    │
                                                    ▼
User (WhatsApp) ◄── WhatsApp Cloud API ◄── Send Message API
```

### 5.2 Command Protocol (WhatsApp)

```
/status              → System health overview
/agents              → List active agents and their status
/task <description>  → Create new task for Commander
/deploy <env>        → Trigger deployment (requires confirmation)
/security            → Latest security scan results
/logs <agent> [n]    → Last n log entries for agent
/approve <task-id>   → Approve pending operation
/reject <task-id>    → Reject pending operation
/cost                → Current API usage and costs
/pause <agent>       → Pause an agent
/resume <agent>      → Resume a paused agent
```

Natural language is also supported — Commander parses intent.

### 5.3 Inter-Agent Communication (A2A Protocol)

```python
# Agent-to-Agent message envelope
{
    "protocol": "a2a/1.0",
    "message_id": "uuid-v4",
    "from_agent": "builder-001",
    "to_agent": "inspector-001",
    "timestamp": "2026-02-06T20:45:00Z",
    "type": "task_request",
    "payload": {
        "task": "run_tests",
        "context": {
            "repository": "prooflog/backend",
            "branch": "feature/auth",
            "commit": "abc123"
        },
        "priority": "high",
        "deadline": "2026-02-06T21:00:00Z"
    },
    "signature": "ed25519:..."
}
```

---

## 6. Data Flow Examples

### 6.1 Feature Development Flow

```
User (WhatsApp): "Add rate limiting to the Prooflog API"
                              │
                              ▼
Commander decomposes into tasks:
  1. Scout: Research best rate limiting patterns for blockchain APIs
  2. Builder: Implement rate limiting middleware
  3. Inspector: Write and run tests
  4. Sentinel: Security review of implementation
  5. Scribe: Update API documentation
  6. DevOps: Deploy to staging
                              │
                              ▼
Commander manages workflow, reports progress via WhatsApp
User approves deployment → DevOps deploys
```

### 6.2 Security Incident Flow

```
Watcher detects anomaly → Alerts Sentinel
                              │
                              ▼
Sentinel analyzes threat:
  • Scans logs for attack patterns
  • Checks for compromised dependencies
  • Evaluates severity (CRITICAL/HIGH/MEDIUM/LOW)
                              │
                              ▼
If CRITICAL:
  → Immediate WhatsApp alert to user
  → Auto-apply mitigation (if pre-approved)
  → Builder patches vulnerability
  → Inspector validates fix
  → DevOps deploys hotfix
```

---

## 7. Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Language | Python 3.12+ | Best AI/LLM ecosystem, Pydantic AI, async-first |
| Agent Framework | Pydantic AI | Type-safe, multi-model, Temporal-ready |
| API Gateway | FastAPI | Async, auto-docs, Pydantic integration |
| Message Bus (fast) | Redis 7+ | Pub/sub, streams, caching, sessions |
| Task Queue (durable) | RabbitMQ | Complex routing, durability, dead-letter queues |
| Database | PostgreSQL 16 | State, audit, vector (pgvector) |
| Cache/Sessions | Redis | Sub-millisecond access |
| Container Runtime | Docker + Compose | Agent isolation, reproducibility |
| Cloud LLM | Anthropic Claude API | Best reasoning, tool use, safety |
| Local LLM | Ollama | Privacy, cost optimization |
| WhatsApp | WhatsApp Cloud API | Official Meta API, webhooks |
| Secrets | HashiCorp Vault (or SOPS) | Secure secret management |
| Monitoring | Prometheus + Grafana | Industry standard |
| CI/CD | GitHub Actions | Prooflog's likely platform |
| Future Orchestration | Temporal | Durable workflows, human-in-the-loop |

---

## 8. Project Structure

```
code-horde/
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md         # This file
│   ├── SECURITY.md             # Security policies
│   └── AGENTS.md               # Agent specifications
├── src/
│   ├── core/                   # Core framework
│   │   ├── __init__.py
│   │   ├── agent_base.py       # Base agent class
│   │   ├── orchestrator.py     # Commander logic
│   │   ├── message_bus.py      # Redis pub/sub + RabbitMQ
│   │   ├── task_manager.py     # Task lifecycle
│   │   └── config.py           # Configuration management
│   ├── agents/                 # Specialized agents
│   │   ├── __init__.py
│   │   ├── sentinel.py         # Security agent
│   │   ├── builder.py          # Development agent
│   │   ├── inspector.py        # QA agent
│   │   ├── watcher.py          # Monitoring agent
│   │   ├── scout.py            # Research agent
│   │   ├── scribe.py           # Documentation agent
│   │   └── devops.py           # Infrastructure agent
│   ├── bridges/                # External communication
│   │   ├── __init__.py
│   │   ├── whatsapp.py         # WhatsApp Cloud API bridge
│   │   └── webhook_handler.py  # Webhook processing
│   ├── security/               # Security subsystem
│   │   ├── __init__.py
│   │   ├── auth.py             # Authentication/Authorization
│   │   ├── capabilities.py     # Capability-based permissions
│   │   ├── audit.py            # Immutable audit trail
│   │   ├── secrets.py          # Secret management
│   │   ├── sandbox.py          # Execution sandboxing
│   │   └── threat_detector.py  # Prompt injection defense
│   ├── models/                 # LLM integration
│   │   ├── __init__.py
│   │   ├── router.py           # Multi-model routing
│   │   ├── claude_client.py    # Anthropic API client
│   │   ├── ollama_client.py    # Ollama local client
│   │   └── schemas.py          # Pydantic models
│   ├── storage/                # Persistence
│   │   ├── __init__.py
│   │   ├── database.py         # PostgreSQL connection
│   │   ├── redis_store.py      # Redis operations
│   │   └── migrations/         # Alembic migrations
│   └── protocols/              # Communication protocols
│       ├── __init__.py
│       ├── a2a.py              # Agent-to-Agent protocol
│       └── mcp_adapter.py      # MCP compatibility
├── config/                     # Configuration files
│   ├── agents/                 # Agent capability manifests
│   │   ├── commander.yaml
│   │   ├── sentinel.yaml
│   │   ├── builder.yaml
│   │   └── ...
│   ├── settings.yaml           # System settings
│   └── security_policies.yaml  # Security rules
├── docker/                     # Docker configuration
│   ├── Dockerfile.agent        # Base agent image
│   ├── Dockerfile.gateway      # API gateway image
│   └── docker-compose.yaml     # Full system orchestration
├── scripts/                    # Utility scripts
│   ├── setup.sh                # Initial setup
│   ├── generate_certs.sh       # TLS certificate generation
│   └── health_check.sh         # System health verification
├── tests/                      # Test suite
│   ├── unit/
│   └── integration/
├── pyproject.toml              # Python project configuration
├── .env.example                # Environment variables template
└── README.md                   # Quick start guide
```

---

## 9. Deployment Roadmap

### Phase 1: Foundation (Weeks 1-2)
- Core framework (agent base, orchestrator, message bus)
- Security subsystem (auth, capabilities, audit trail)
- Multi-model router (Claude + Ollama)
- WhatsApp bridge (basic commands)
- Docker containerization

### Phase 2: Agent Army (Weeks 3-4)
- Implement all 8 specialized agents
- Inter-agent communication (A2A protocol)
- Task workflow engine
- Prooflog-specific integrations

### Phase 3: Intelligence (Weeks 5-6)
- Advanced routing (cost/sensitivity/complexity)
- Context management and memory
- Learning from past actions
- Anomaly detection

### Phase 4: Cloud Command Center (Weeks 7-8+)
- Web dashboard (React/Next.js)
- Cloud deployment (Kubernetes)
- Multi-tenant support
- Real-time monitoring dashboard
- Mobile companion app

---

## 10. Future: Cloud Command Center

```
┌─────────────────────────────────────────────────────────────┐
│                   Cloud Command Center                       │
│                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│   │  Dashboard    │  │  Agent Fleet │  │  Analytics       │ │
│   │  (React)      │  │  Manager     │  │  Engine          │ │
│   └──────────────┘  └──────────────┘  └──────────────────┘ │
│                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│   │  Multi-Tenant │  │  Billing     │  │  Marketplace     │ │
│   │  Isolation    │  │  (Usage)     │  │  (Agent Plugins) │ │
│   └──────────────┘  └──────────────┘  └──────────────────┘ │
│                                                              │
│   Connects to local Code Horde instances via secure tunnels   │
│   (WireGuard / Cloudflare Tunnel / Tailscale)               │
└─────────────────────────────────────────────────────────────┘
```
