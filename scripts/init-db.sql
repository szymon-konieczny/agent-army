-- ============================================================================
-- Code Horde — Database Initialization
-- ============================================================================
-- Creates core tables for the agent system.
-- Runs automatically on first PostgreSQL container start.
-- ============================================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- Audit Trail (immutable, hash-chained)
-- ============================================================================
CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL PRIMARY KEY,
    entry_id        UUID NOT NULL DEFAULT uuid_generate_v4() UNIQUE,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_id        TEXT NOT NULL,
    action          TEXT NOT NULL,
    target          TEXT,
    input_hash      TEXT,
    output_hash     TEXT,
    prev_hash       TEXT,         -- hash of previous entry → chain
    entry_hash      TEXT NOT NULL, -- hash of this entry
    signature       TEXT,          -- agent's Ed25519 signature
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_agent    ON audit_log (agent_id);
CREATE INDEX idx_audit_action   ON audit_log (action);
CREATE INDEX idx_audit_ts       ON audit_log (timestamp DESC);
CREATE INDEX idx_audit_target   ON audit_log (target);

-- ============================================================================
-- Tasks
-- ============================================================================
CREATE TABLE IF NOT EXISTS tasks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    description     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','assigned','in_progress',
                                      'completed','failed','cancelled')),
    priority        INT NOT NULL DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),
    assigned_agent  TEXT,
    payload         JSONB DEFAULT '{}',
    result          JSONB,
    error           TEXT,
    tags            TEXT[] DEFAULT '{}',
    parent_task_id  UUID REFERENCES tasks(id),
    timeout_seconds INT DEFAULT 3600,
    max_retries     INT DEFAULT 3,
    retry_count     INT DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ
);

CREATE INDEX idx_tasks_status   ON tasks (status);
CREATE INDEX idx_tasks_agent    ON tasks (assigned_agent);
CREATE INDEX idx_tasks_priority ON tasks (priority DESC, created_at ASC);
CREATE INDEX idx_tasks_parent   ON tasks (parent_task_id);

-- ============================================================================
-- Agent Registry
-- ============================================================================
CREATE TABLE IF NOT EXISTS agents (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    role            TEXT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'offline'
                    CHECK (state IN ('idle','busy','paused','error','offline')),
    capabilities    JSONB DEFAULT '[]',
    public_key      TEXT,
    last_heartbeat  TIMESTAMPTZ,
    tasks_completed BIGINT DEFAULT 0,
    tasks_failed    BIGINT DEFAULT 0,
    metadata        JSONB DEFAULT '{}',
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- Conversation History (WhatsApp + internal)
-- ============================================================================
CREATE TABLE IF NOT EXISTS conversations (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source          TEXT NOT NULL CHECK (source IN ('whatsapp','api','internal','webhook')),
    external_id     TEXT,          -- WhatsApp message ID, etc.
    sender          TEXT NOT NULL,
    content         TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('inbound','outbound')),
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_conv_source    ON conversations (source, created_at DESC);
CREATE INDEX idx_conv_sender    ON conversations (sender);

-- ============================================================================
-- Security Events
-- ============================================================================
CREATE TABLE IF NOT EXISTS security_events (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    severity        TEXT NOT NULL CHECK (severity IN ('low','medium','high','critical')),
    event_type      TEXT NOT NULL,
    agent_id        TEXT,
    description     TEXT NOT NULL,
    details         JSONB DEFAULT '{}',
    resolved        BOOLEAN DEFAULT FALSE,
    resolved_at     TIMESTAMPTZ,
    resolved_by     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_secevt_sev     ON security_events (severity, created_at DESC);
CREATE INDEX idx_secevt_type    ON security_events (event_type);
CREATE INDEX idx_secevt_unresolved ON security_events (resolved) WHERE resolved = FALSE;

-- ============================================================================
-- Trigger: auto-update updated_at
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_tasks_updated
    BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_agents_updated
    BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
