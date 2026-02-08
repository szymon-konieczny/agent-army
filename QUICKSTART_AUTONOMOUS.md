# Quick Start: Autonomous Operations

Get up and running with autonomous operations in 5 minutes.

## Installation

Files are already created in your project. No additional dependencies needed beyond what's in `pyproject.toml`.

## Minimal Setup (Copy-Paste Ready)

```python
import asyncio
from src.security.policy_engine import PolicyEngine
from src.security.trust_score import TrustScorer
from src.security.guardrails import GuardrailRunner, CostGuardrail, SecretLeakGuardrail
from src.security.autonomous_operations import AutonomousOperations, ActionRequest
from src.bridges.whatsapp_digest import WhatsAppDigest, DigestConfig


async def main():
    # 1. Initialize components
    policy_engine = PolicyEngine()
    policy_engine.load_default_policies()  # Uses sensible defaults

    trust_scorer = TrustScorer()

    guardrail_runner = GuardrailRunner()
    guardrail_runner.register(CostGuardrail(warn_threshold=1.0, error_threshold=10.0))
    guardrail_runner.register(SecretLeakGuardrail())

    digest = WhatsAppDigest(config=DigestConfig(mode="digest"))
    # digest.set_send_function(your_whatsapp_send_fn)  # Optional for testing

    # 2. Create coordinator
    ops = AutonomousOperations(
        policy_engine=policy_engine,
        trust_scorer=trust_scorer,
        guardrail_runner=guardrail_runner,
        digest=digest,
    )

    # 3. Start background tasks
    await ops.start_background_tasks()

    # 4. Test: Agent wants to run tests (should be GREEN)
    request = ActionRequest(
        agent_id="builder-1",
        action="test.run_suite",
        parameters={"directory": "tests/"},
    )

    result = await ops.request_action(request)
    print(f"✓ Action executed: tier={result.tier}, status={result.status}")
    # Output: ✓ Action executed: tier=green, status=success

    # 5. Test: Agent wants to deploy to prod (should be RED)
    request = ActionRequest(
        agent_id="devops-1",
        action="deploy.production",
        parameters={"version": "1.2.3"},
    )

    result = await ops.request_action(request)
    print(f"✓ Approval needed: tier={result.tier}, status={result.status}")
    # Output: ✓ Approval needed: tier=red, status=waiting_for_approval

    # 6. Simulate approval
    pending_ids = list(ops.pending_approvals.keys())
    if pending_ids:
        request_id = pending_ids[0]
        final_result = await ops.approve_action(request_id, approved=True)
        print(f"✓ Action executed: status={final_result.status}")

    # 7. Check agent status
    status = ops.get_agent_status("builder-1")
    print(f"✓ Builder trust: {status['trust_level']} ({status['trust_score']:.0f}/100)")

    # 8. Cleanup
    await ops.stop_background_tasks()


if __name__ == "__main__":
    asyncio.run(main())
```

## Run It

```bash
cd /sessions/quirky-charming-cori/mnt/agent-army
python3 -m asyncio quickstart.py  # Or save above as quickstart.py
```

## What Just Happened

1. **Policy Engine** classified actions:
   - `test.run_suite` → GREEN (autonomous)
   - `deploy.production` → RED (needs approval)

2. **Trust Scorer** updated agent reputation:
   - Successful test run: +1.0 point
   - Approved deploy: +1.0 point

3. **Guardrails** ran automatic checks:
   - Cost OK (no expensive LLM calls)
   - No secrets in payload
   - Rate limits not exceeded

4. **Digest** collected YELLOW actions for batching

5. **Workflow Engine** is ready for multi-step tasks

## Next Steps

### 1. Customize Policies

Edit `/config/autonomy_policies.yaml`:

```yaml
tiers:
  green:
    actions:
      - "test.run*"
      - "code.write"  # But not .env, secrets/
      - "git.read*"

  yellow:
    actions:
      - "git.create_pull_request"
      - "deploy.staging"

  red:
    actions:
      - "deploy.production"

  black:
    actions:
      - "secrets.expose"
```

Reload:
```python
ops.policy_engine.load_policies("config/autonomy_policies.yaml")
```

### 2. Add WhatsApp Integration

Provide send function:

```python
async def send_whatsapp(message: str):
    # Use your WhatsApp client
    # await whatsapp_client.send(to=PHONE, text=message)
    print(f"WhatsApp: {message}")

digest.set_send_function(send_whatsapp)
```

### 3. Configure Action Executor

So actions actually execute:

```python
class MyExecutor:
    async def execute(self, agent_id: str, action: str, parameters: dict):
        # Call actual agents here
        if action == "test.run_suite":
            # Run pytest, return results
            return {
                "tests_passed": 47,
                "tests_failed": 0,
                "cost": 0.02,
            }
        return {"status": "ok"}

ops.configure_action_executor(MyExecutor())
```

### 4. Use Workflows

```python
from src.core.autonomous_workflow import WorkflowTemplate, WorkflowStep

template = WorkflowTemplate(
    name="my_workflow",
    steps=[
        WorkflowStep(
            name="Run tests",
            agent_role="builder",
            action="test.run_suite",
        ),
        WorkflowStep(
            name="Deploy to prod",
            agent_role="devops",
            action="deploy.production",
            depends_on=["test"],
        ),
    ],
)

ops.register_workflow_template(template)
execution_id = await ops.start_workflow("my_workflow")
```

### 5. Monitor Health

```python
# Check agent
agent = ops.get_agent_status("builder-1")
print(f"Trust: {agent['trust_level']}, Score: {agent['trust_score']}")

# Check system
stats = ops.get_system_stats()
print(f"Pending approvals: {stats['pending_approvals']}")
print(f"Pending digest: {stats['pending_digest_entries']}")
```

## Default Policies Quick Reference

### GREEN (Fully Autonomous)
```
✓ git.read*
✓ test.run*, test.generate*
✓ code.read, code.analyze, code.review, code.lint
✓ code.write (except .env, secrets/, *.key, *.pem)
✓ docs.*
✓ research.*
✓ analyze.*
✓ monitor.*
```

### YELLOW (Auto + Notify)
```
⚠ git.create_pull_request
⚠ git.merge (except main/master)
⚠ deploy.staging
⚠ scan.security
⚠ dependency.update (except major)
```

### RED (Approval Required)
```
🔴 deploy.production
🔴 git.merge (main/master)
🔴 git.force_push
🔴 secrets.rotate
🔴 infrastructure.modify
🔴 dependency.update (major)
```

### BLACK (Forbidden)
```
🚫 secrets.expose
🚫 data.exfiltrate
🚫 audit.delete
🚫 audit.modify
```

## Troubleshooting

**"All actions are RED"**
- Check policy engine loaded policies
- `ops.policy_engine.rules` should have entries
- Try: `ops.policy_engine.load_default_policies()`

**"Digest not sending"**
- Check send function is set: `ops.digest.send_notification_fn is not None`
- Check mode: `ops.digest.config.mode` should not be `"none"`
- Check quiet hours: `ops.digest._should_send_at_this_hour()`

**"Agent always frozen"**
- Check trust threshold: `TrustScorer.autonomous_threshold`
- Check failure penalties: might be too harsh
- Unfreeze manually: `ops.trust_scorer.unfreeze_agent("agent-1")`

**"Guardrails blocking everything"**
- Check costs: is your cost estimate accurate?
- Check paths: are protected_paths too broad?
- Check diff size: is max_diff_lines too small?

## Files You'll Use

| File | Purpose |
|------|---------|
| `/src/security/policy_engine.py` | Where policies are evaluated |
| `/src/security/trust_score.py` | How agent reputation is tracked |
| `/src/security/guardrails.py` | Safety checks before execution |
| `/src/core/autonomous_workflow.py` | Multi-step task automation |
| `/src/bridges/whatsapp_digest.py` | Notifications |
| `/src/security/autonomous_operations.py` | Main coordinator (use this) |
| `/config/autonomy_policies.yaml` | Edit this to customize |

## Architecture at a Glance

```
Agent wants to do something
         ↓
request_action(ActionRequest)
         ↓
┌─ PolicyEngine.evaluate()
│  ├─ Find matching rule
│  ├─ Check trust score
│  ├─ Apply context conditions
│  └─ Return tier (GREEN/YELLOW/RED/BLACK)
│
├─ GuardrailRunner.run_all()
│  ├─ CostGuardrail ✓
│  ├─ SecretLeakGuardrail ✓
│  ├─ BranchProtectionGuardrail ✓
│  └─ ...7 checks total
│
├─ BLACK? → Reject, done
├─ RED? → Send approval request, pause
├─ GREEN/YELLOW? → Execute
│
├─ Executor.execute()
│
├─ TrustScorer.record_success/failure()
│  └─ Update agent reputation
│
├─ Digest.add_event()
│  └─ Queue for notification
│
└─ Return ActionResult
```

## Real-World Example

Your agent wants to create a PR:

```python
request = ActionRequest(
    agent_id="builder-1",
    action="git.create_pull_request",
    parameters={
        "branch": "feature/rate-limit",
        "title": "Add rate limiting to API",
        "description": "...",
    },
    estimated_cost=0.15,
)

result = await ops.request_action(request)
# ✓ Evaluated as YELLOW (autonomous + notify)
# ✓ Guardrails pass (cost OK, no secrets, safe files)
# ✓ Executor.execute() called
# ✓ Trust increased: 50 → 51 (+1 success)
# ✓ DigestEntry queued
# ✓ In 30 minutes, user gets summary:
#   "Builder created PR #42: Add rate limiting"
```

## Key Concepts

**Tier:** What tier an action is classified as (GREEN/YELLOW/RED/BLACK)
**Trust:** Agent's reputation score (0-100) that can elevate tiers
**Guardrail:** Automated safety check (cost, secrets, rate limit, etc.)
**Workflow:** Multi-step task like "deploy to production"
**Digest:** Batched notification instead of alert spam

## Need More?

- **Full Guide:** `/AUTONOMOUS_OPERATIONS_GUIDE.md`
- **Manifest:** `/AUTONOMOUS_SYSTEM_MANIFEST.md`
- **Config:** `/config/autonomy_policies.yaml`
- **Code:** Check docstrings in each module

Good luck! 🚀
