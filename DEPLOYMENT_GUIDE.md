# AgentArmy Deployment Guide

## Overview

This document provides a comprehensive guide to the AgentArmy project's Docker and configuration files created for deployment and orchestration of intelligent agents.

## Project Structure

```
agent-army/
├── docker/
│   ├── Dockerfile.agent          # Base agent container image
│   ├── Dockerfile.gateway        # API gateway container image
│   └── docker-compose.yaml       # Full system orchestration
├── config/
│   ├── settings.yaml             # System-wide configuration
│   ├── security_policies.yaml    # Security rules and rate limits
│   └── agents/
│       ├── commander.yaml        # Commander agent capabilities
│       ├── sentinel.yaml         # Sentinel security agent
│       ├── builder.yaml          # Builder development agent
│       ├── inspector.yaml        # Inspector testing agent
│       └── watcher.yaml          # Watcher monitoring agent
├── scripts/
│   └── setup.sh                  # Automated setup script
├── .env.example                  # Environment variables template
└── requirements.txt              # Python dependencies
```

## Files Created

### 1. Docker Images

#### docker/Dockerfile.agent (53 lines)
**Purpose:** Base image for all agent containers

**Features:**
- Multi-stage build for minimal layer size
- Python 3.12-slim base image
- Non-root user (agentuser:1000) for security
- Pre-built Python dependencies cached
- Health check endpoint on port 9000
- Proper signal handling with exec form entrypoint

**Key Sections:**
```dockerfile
FROM python:3.12-slim as builder
FROM python:3.12-slim
RUN useradd -m -u 1000 agentuser
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3
ENTRYPOINT ["python", "-m"]
CMD ["agent.worker"]
```

#### docker/Dockerfile.gateway (53 lines)
**Purpose:** API Gateway container for external access

**Features:**
- Multi-stage build optimization
- Exposes port 8000 for FastAPI
- Non-root gatewayuser:1000
- UV Icorn server with auto-reload capability
- Health checks for API availability

**Key Sections:**
```dockerfile
FROM python:3.12-slim as builder
EXPOSE 8000
USER gatewayuser
CMD ["uvicorn", "src.gateway.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Docker Compose Orchestration

#### docker/docker-compose.yaml (323 lines)
**Purpose:** Complete system orchestration with 9 services

**Infrastructure Services:**
- **redis:7-alpine** - Caching and session management
  - Port: 6379
  - Volume: redis-data
  - Healthcheck: redis-cli ping

- **rabbitmq:3-management-alpine** - Message queue
  - Ports: 5672 (AMQP), 15672 (Management UI)
  - Volume: rabbitmq-data
  - Default credentials configurable via ENV

- **postgres:16-alpine** - Primary database
  - Port: 5432
  - Volume: postgres-data
  - Automatic user/DB creation

**Agent Services** (6 containers):
1. **gateway** - API entry point
   - CPU: 1-2 cores
   - Memory: 1-2GB
   - Networks: agent-net, gateway-net

2. **commander** - Orchestration control
   - CPU: 1-2 cores, Memory: 1-2GB
   - Highest priority (start first)
   - Full system access

3. **sentinel** - Security monitoring
   - CPU: 0.75-1.5 cores, Memory: 750M-1.5GB
   - Read-only access to most systems
   - Continuous threat detection

4. **builder** - Development and builds
   - CPU: 1.5-3 cores, Memory: 2-4GB
   - Docker socket access for image building
   - GitHub token access

5. **inspector** - Testing and QA
   - CPU: 1-2 cores, Memory: 1-2GB
   - Read-only code access
   - Test artifact generation

6. **watcher** - Monitoring and logging
   - CPU: 0.5-1 core, Memory: 500M-1GB
   - Log aggregation
   - Metrics collection

**Networking:**
- **agent-net** (internal) - Secure inter-agent communication
- **gateway-net** (external) - Public API access

**Resource Management:**
- CPU requests/limits per container
- Memory quotas (500MB-4GB per agent)
- Storage volumes for persistence
- Restart policies: unless-stopped

### 3. System Configuration

#### config/settings.yaml (246 lines)
**Purpose:** Central system configuration

**Sections:**
- **System metadata** - Name, version, environment
- **Agent definitions** - 5 agents with roles, priorities, capabilities
- **LLM routing** - Multi-model support (Claude 3.5 Sonnet, Opus 4.6, Ollama)
- **WhatsApp integration** - Command definitions and message templates
- **Monitoring intervals** - Health checks, metrics, log aggregation
- **Message queue configuration** - RabbitMQ exchanges and routing
- **Cache configuration** - Redis TTL and key prefixes
- **Database pooling** - PostgreSQL connection pool setup
- **API configuration** - Host, port, CORS settings
- **Security settings** - JWT, Ed25519 keys, request signing
- **Logging configuration** - Format, output, per-agent levels

#### config/security_policies.yaml (408 lines)
**Purpose:** Comprehensive security governance

**Sections:**
- **Token limits** - Per-agent daily/hourly quotas
  - Commander: 1M tokens/day
  - Sentinel: 500K tokens/day
  - Builder: 800K tokens/day
  - Inspector: 600K tokens/day
  - Watcher: 400K tokens/day

- **Rate limits** - Action-specific restrictions
  - Git operations: push, branch, merge limits
  - Deployment: 10 per hour max
  - Build operations: concurrent limits
  - WhatsApp: 1000 per day

- **Forbidden actions** - Critical, high, medium severity
  - Critical: Database deletion, audit log deletion
  - High: Force push to main, firewall modification
  - Medium: Modify another agent's config

- **Escalation rules** - What triggers human approval
  - Deploy to production: 2 approvals required
  - Force push: 1 approval required
  - Security anomaly: automatic Sentinel escalation

- **Network access control** - By agent
  - Commander: All hosts, all ports
  - Builder: GitHub, PyPI, Docker, localhost
  - Sentinel: Only localhost/internal

- **Database access control** - Read/write permissions
- **Audit logging** - Field redaction, retention policies
- **Encryption** - AES-256-GCM for sensitive data

### 4. Agent Configurations

#### config/agents/commander.yaml (178 lines)
**Purpose:** Master orchestrator configuration

**Capabilities:**
- Agent lifecycle (start, stop, restart, scale)
- Workflow orchestration
- Full message queue access
- All LLM models
- WhatsApp messaging
- Emergency override functions
- System administration

**Approvals Required:** None (full autonomy)

**Resource Limits:**
- CPU: 1-2 cores
- Memory: 1-2GB
- Max tokens: 1,000,000/day

#### config/agents/sentinel.yaml (236 lines)
**Purpose:** Security monitoring and threat detection

**Capabilities:**
- Repository read-only access
- System monitoring (read-only)
- Security log writing
- Filesystem scanning
- Network monitoring
- Alert triggering
- Threat analysis via LLM

**Key Restrictions:**
- No write access to code
- Cannot delete or force push
- Forbidden paths: /root, /etc/shadow

**Resource Limits:**
- CPU: 0.75-1.5 cores
- Memory: 750M-1.5GB
- Max tokens: 500,000/day

#### config/agents/builder.yaml (300 lines)
**Purpose:** Development and build orchestration

**Capabilities:**
- Git read/write (with restrictions)
- Filesystem access (src, tests only)
- Build execution (make, docker, npm, pip)
- Test execution
- Artifact management
- Docker image building
- Network access (GitHub, PyPI, NPM)

**Forbidden Actions:**
- Force push to main/production
- Delete main/production branches
- Read /root, /etc, PostgreSQL files

**Requires Approval For:**
- Deploy to production/staging
- Modify CI/CD pipeline
- Delete branch
- Force push
- Public artifact publishing

**Resource Limits:**
- CPU: 1.5-3 cores
- Memory: 2-4GB
- Max tokens: 800,000/day

#### config/agents/inspector.yaml (249 lines)
**Purpose:** Quality assurance and testing

**Capabilities:**
- Test execution (pytest, jest, mocha, etc.)
- Code analysis (lint, security, complexity)
- Report generation (JSON, HTML, XML, CSV)
- Coverage analysis
- Dependency analysis
- LLM-powered test analysis

**Key Features:**
- Parallel test execution (up to 5 concurrent)
- Coverage threshold enforcement (80% minimum)
- Quality gates (test passing, linting, security)
- Automatic test retries (up to 2x)

**Resource Limits:**
- CPU: 1-2 cores
- Memory: 1-2GB
- Max tokens: 600,000/day

#### config/agents/watcher.yaml (334 lines)
**Purpose:** System monitoring, logging, and alerting

**Capabilities:**
- Log aggregation and rotation
- Metrics collection (CPU, memory, disk, network)
- System monitoring and health checks
- Anomaly detection
- Alert notification (WhatsApp, email, Slack)
- Dashboard generation
- Data export (JSON, CSV, Parquet)

**Monitoring Parameters:**
- CPU threshold: 80%
- Memory threshold: 85%
- Disk threshold: 90%
- Metrics collection: 60s interval
- Metrics retention: 90 days

**Scheduled Jobs:**
- Hourly metrics collection
- Daily log rotation
- Weekly log archival
- Daily health summary
- 5-minute anomaly detection

**Resource Limits:**
- CPU: 0.5-1 core
- Memory: 500M-1GB
- Max tokens: 400,000/day

### 5. Environment Configuration

#### .env.example (105 lines)
**Purpose:** Template for environment variables

**Categories:**

**System Configuration:**
- AGENT_ENV (development/staging/production)
- LOG_LEVEL (DEBUG/INFO/WARNING/ERROR)

**Database:**
- DB_USER, DB_PASS, DB_NAME
- DATABASE_URL (full connection string)

**Cache & Queue:**
- REDIS_URL
- RABBITMQ_USER, RABBITMQ_PASS, RABBITMQ_URL

**Security:**
- JWT_SECRET (64-char hex)
- AGENT_PRIVATE_KEYS (Ed25519 format, 5 agents)

**API Keys:**
- CLAUDE_API_KEY (Anthropic)
- GITHUB_TOKEN (GitHub PAT)

**WhatsApp:**
- WHATSAPP_TOKEN
- WHATSAPP_BUSINESS_ACCOUNT_ID
- WHATSAPP_PHONE_NUMBER_ID

**Optional Services:**
- SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET
- SMTP configuration (Gmail)
- OLLAMA_BASE_URL
- SENTRY_DSN

**Feature Flags:**
- ENABLE_WHATSAPP
- ENABLE_SLACK
- ENABLE_EMAIL
- ENABLE_OLLAMA
- ENABLE_SECURITY_SCANNING

### 6. Setup Script

#### scripts/setup.sh (378 lines)
**Purpose:** Automated initialization and configuration

**Functions:**

1. **check_prerequisites()** - Verifies installed tools
   - Docker & Docker Compose
   - Python 3
   - OpenSSL
   - Git

2. **create_env_file()** - Generates .env from template
   - Copies .env.example to .env
   - Sets file permissions to 600

3. **generate_jwt_secret()** - Creates 64-char hex JWT secret
   - Uses openssl rand -hex 32
   - Updates .env file

4. **generate_agent_keys()** - Creates Ed25519 key pairs
   - Generates 5 key pairs (one per agent)
   - Stores in keys/ directory
   - Updates .env with private keys
   - Maintains proper permissions (600 private, 644 public)

5. **setup_docker_network()** - Creates Docker networks
   - agent-army-internal (internal)
   - agent-army-gateway (external)

6. **create_directories()** - Creates project directories
   - logs, data, artifacts, keys, backups

7. **validate_configuration()** - Checks configuration validity
   - JWT_SECRET format
   - Agent key presence
   - API key configuration

8. **print_success_message()** - Displays setup completion
   - Lists next steps
   - Shows usage commands
   - Security warnings

**Usage:**
```bash
bash scripts/setup.sh
```

**Output:**
- Color-coded logging (INFO, SUCCESS, WARNING, ERROR)
- Progress tracking
- Detailed completion report

## Deployment Steps

### Prerequisites
- Docker (20.10+)
- Docker Compose (2.0+)
- Python 3.9+
- OpenSSL
- Git

### Step 1: Run Setup Script
```bash
cd /sessions/quirky-charming-cori/mnt/agent-army
bash scripts/setup.sh
```

### Step 2: Configure Environment Variables
```bash
# Edit .env with your specific values
nano .env

# Required values:
# - CLAUDE_API_KEY
# - GITHUB_TOKEN
# - WHATSAPP_TOKEN
# - Database credentials
# - API keys for any external services
```

### Step 3: Verify Configuration
```bash
# Validate YAML files
python3 -m yaml < config/settings.yaml > /dev/null

# Check .env
grep -E "^[A-Z_]+=.+" .env | head -20
```

### Step 4: Start Services
```bash
# Start all containers in background
docker-compose up -d

# Watch startup logs
docker-compose logs -f gateway

# Check service health
docker-compose ps
```

### Step 5: Verify Deployment
```bash
# Check API gateway is responding
curl http://localhost:8000/health

# Verify agent health
docker-compose exec commander python -c "import socket; socket.create_connection(('localhost', 9000))"

# Check RabbitMQ management UI
# Visit: http://localhost:15672 (guest:guest)

# Check Redis connectivity
docker-compose exec redis redis-cli ping
```

### Step 6: View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f builder

# Follow error logs only
docker-compose logs -f --until=5m | grep -i error
```

## Key Architectural Decisions

### 1. Multi-Stage Docker Builds
- Reduces final image size by excluding build dependencies
- Faster deployments and reduced storage usage
- Separate caching of dependencies from application code

### 2. Non-Root Users
- Each container runs as non-root (agentuser, gatewayuser)
- Improves security by limiting privilege escalation

### 3. Dual Network Setup
- **agent-net (internal)**: Isolated inter-agent communication
- **gateway-net (external)**: Controlled external access
- Prevents agents from accidentally exposing sensitive endpoints

### 4. Resource Limits
- CPU and memory quotas prevent resource exhaustion
- Prevents single agent from impacting entire system
- Enables predictable scaling

### 5. Comprehensive Security Model
- Role-based access control per agent
- Token usage limits prevent excessive API calls
- Rate limiting on sensitive operations
- Forbidden action lists prevent dangerous operations
- Approval workflows for high-risk actions

### 6. Observability
- Health checks on all containers
- Structured logging with redaction
- Metrics collection for all system components
- Centralized log aggregation

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs gateway

# Rebuild image
docker-compose build --no-cache

# Remove and restart
docker-compose down
docker-compose up -d
```

### Health Check Failures
```bash
# Check service health
docker-compose ps

# Manual health check
docker-compose exec gateway curl http://localhost:8000/health

# Increase startup grace period
# Edit docker-compose.yaml: start_period in healthcheck
```

### Database Connection Issues
```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Check database status
docker-compose exec postgres psql -U agentadmin -d agent_army -c "SELECT 1"

# View connection logs
docker-compose logs postgres
```

### Rate Limit or Token Issues
```bash
# Check agent token usage
docker-compose logs watcher | grep -i token

# Review security policies
cat config/security_policies.yaml | grep -A 5 "token_limits"
```

## Maintenance

### Backup Strategy
```bash
# Backup volumes
docker-compose exec postgres pg_dump -U agentadmin agent_army > backup.sql

# Backup Redis
docker-compose exec redis redis-cli BGSAVE

# Backup RabbitMQ
docker-compose exec rabbitmq rabbitmq-diagnostics export_definitions
```

### Updates and Upgrades
```bash
# Update images
docker-compose pull

# Rebuild with new images
docker-compose build --no-cache

# Restart services with zero downtime
docker-compose up -d --no-deps --build gateway
```

### Log Rotation
```bash
# View Watcher logs about rotation
docker-compose logs watcher | grep -i "rotate"

# Manual cleanup
docker-compose exec watcher rm -rf /var/log/old
```

## Security Considerations

1. **Secret Management**
   - Store JWT_SECRET and private keys securely
   - Never commit .env to version control
   - Rotate keys quarterly

2. **Network Isolation**
   - agent-net is internal only
   - Only gateway is exposed externally
   - All inter-agent communication encrypted

3. **Access Control**
   - Each agent has specific capabilities
   - No agent can access another's private keys
   - Read-only access where possible

4. **Audit Logging**
   - All actions logged with timestamps
   - Sensitive fields redacted automatically
   - 1-year retention for critical actions

5. **Regular Scanning**
   - Sentinel runs continuous security scans
   - Dependency vulnerability checks
   - Network anomaly detection

## Performance Tuning

### Database
```yaml
# postgres service in docker-compose.yaml
- shared_buffers=256MB
- effective_cache_size=1GB
```

### Cache
```yaml
# redis service
redis-server --maxmemory 512mb
```

### Message Queue
```yaml
# rabbitmq service
RABBITMQ_CHANNEL_MAX=2048
```

## Scaling Considerations

### Horizontal Scaling
```bash
# Run multiple instances of builder agent
docker-compose up -d --scale builder=3
```

### Vertical Scaling
```bash
# Update resource limits in docker-compose.yaml
# Under builder service:
deploy:
  resources:
    limits:
      cpus: '5'
      memory: 8G
```

## Support and Monitoring

### Health Dashboard
- Access Watcher agent metrics via webhook
- Use Watcher's dashboard generation for visualization

### Alert System
- Critical alerts sent via WhatsApp
- High/Medium alerts to Slack
- All alerts logged to PostgreSQL

### Log Query
```bash
# Example: Find all build failures
docker-compose exec postgres psql -U agentadmin -d agent_army \
  -c "SELECT * FROM logs WHERE level='ERROR' AND agent='builder'"
```

## File Statistics

| File | Lines | Purpose |
|------|-------|---------|
| docker/Dockerfile.agent | 53 | Agent base image |
| docker/Dockerfile.gateway | 53 | API gateway image |
| docker/docker-compose.yaml | 323 | System orchestration |
| config/settings.yaml | 246 | System configuration |
| config/security_policies.yaml | 408 | Security governance |
| config/agents/commander.yaml | 178 | Commander config |
| config/agents/sentinel.yaml | 236 | Sentinel config |
| config/agents/builder.yaml | 300 | Builder config |
| config/agents/inspector.yaml | 249 | Inspector config |
| config/agents/watcher.yaml | 334 | Watcher config |
| .env.example | 105 | Environment template |
| scripts/setup.sh | 378 | Setup automation |
| **TOTAL** | **2,863** | **Complete system** |

---

Generated: February 6, 2026
AgentArmy v1.0.0
