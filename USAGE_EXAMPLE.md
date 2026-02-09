# Code Horde Usage Examples

## Basic Setup

### 1. Load Configuration

```python
from src.core.config import MasterSettings

# Load all settings from environment variables and .env file
config = MasterSettings.from_env()

print(f"System: {config.system.system_name}")
print(f"Environment: {config.system.environment}")
print(f"Redis: {config.redis.connection_url}")
print(f"Database: {config.database.connection_url}")
```

### 2. Create and Register Agents

```python
import asyncio
from src.core.agent_base import AgentCapability, AgentIdentity, BaseAgent
from src.core.orchestrator import Orchestrator
from typing import Any

class DataProcessorAgent(BaseAgent):
    """Example agent that processes data."""

    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process a data task."""
        data = task.get("data", [])
        # Do processing...
        return {
            "status": "success",
            "processed_count": len(data),
            "result": f"Processed {len(data)} items"
        }

async def main():
    # Create orchestrator
    orchestrator = Orchestrator(whatsapp_webhook_url="https://api.whatsapp.com/...")

    # Create and register agents
    data_agent = DataProcessorAgent(
        identity=AgentIdentity(
            name="Data Processor",
            role="data_processing",
            capabilities=[
                AgentCapability(
                    name="data_processing",
                    description="Process and transform data",
                    parameters={"batch_size": 100}
                ),
                AgentCapability(
                    name="validation",
                    description="Validate data quality"
                )
            ],
            security_level=3
        )
    )

    await orchestrator.register_agent(data_agent)

    # Discover agents with capability
    available = orchestrator.discover_agents(capability="data_processing")
    print(f"Available agents: {[a.name for a in available]}")
```

### 3. Create and Execute Tasks

```python
from src.core.orchestrator import TaskPriority

async def create_and_execute_task():
    # Create a task
    task = orchestrator.create_task(
        description="Process customer data batch",
        priority=TaskPriority.HIGH,
        payload={
            "data": [
                {"customer_id": 1, "name": "John"},
                {"customer_id": 2, "name": "Jane"}
            ],
            "batch_size": 100
        },
        required_capabilities=["data_processing"],
        timeout_seconds=300,
        max_retries=3
    )

    print(f"Created task: {task.id}")

    # Route task to best agent
    if await orchestrator.route_task(task):
        # Execute the task
        result = await orchestrator.execute_task(task.id)
        print(f"Task result: {result}")

        # Get task status
        completed_task = orchestrator.get_task(task.id)
        print(f"Task status: {completed_task.status}")
    else:
        print("Failed to assign task to any agent")
```

### 4. Workflow Execution

```python
async def execute_workflow_example():
    """Execute multiple tasks in a workflow."""

    # Create sequential workflow
    task1 = orchestrator.create_task(
        description="Extract data",
        priority=TaskPriority.NORMAL,
        required_capabilities=["data_processing"]
    )

    task2 = orchestrator.create_task(
        description="Validate data",
        priority=TaskPriority.NORMAL,
        required_capabilities=["validation"],
        parent_task_id=task1.id  # Depends on task1
    )

    task3 = orchestrator.create_task(
        description="Generate report",
        priority=TaskPriority.NORMAL,
        parent_task_id=task2.id  # Depends on task2
    )

    # Execute sequential workflow
    results = await orchestrator.execute_workflow(
        workflow_name="data_pipeline",
        workflow_type="sequential",
        tasks=[task1, task2, task3]
    )

    print(f"Workflow results: {results}")
```

### 5. Message Bus Communication

```python
from src.core.message_bus import Message, MessageBus, MessageType

async def messaging_example():
    # Create message bus
    msg_bus = MessageBus(
        redis_url=config.redis.connection_url,
        rabbitmq_url=config.rabbitmq.connection_url
    )

    await msg_bus.startup()

    # Publish a command to an agent
    command = Message(
        from_agent="orchestrator",
        to_agent="agent_001",
        message_type=MessageType.COMMAND,
        payload={"action": "pause_processing"}
    )

    await msg_bus.publish(command, durable=False)

    # Send a durable request
    request = Message(
        from_agent="orchestrator",
        to_agent="agent_001",
        message_type=MessageType.REQUEST,
        payload={"query": "What is your current status?"}
    )

    response = await msg_bus.request(request, timeout_seconds=10.0)
    if response:
        print(f"Agent response: {response.payload}")

    await msg_bus.shutdown()
```

### 6. Task Manager

```python
from src.core.task_manager import TaskManager, TaskStatus

async def task_manager_example():
    task_mgr = TaskManager()

    # Create task
    task_id = await task_mgr.create_task(
        description="Process report",
        priority=2,  # HIGH priority
        payload={"report_type": "monthly"},
        max_retries=2,
        tags=["reporting", "monthly"]
    )

    # Assign to agent
    await task_mgr.assign_task(task_id, "agent_001")

    # Update status
    await task_mgr.update_task_status(task_id, TaskStatus.IN_PROGRESS)

    # Complete task
    result = await task_mgr.complete_task(
        task_id,
        {"report_data": {...}, "status": "success"}
    )

    print(f"Task completed in {result.execution_time_seconds} seconds")

    # Get task history (includes retries)
    history = task_mgr.get_task_history(task_id)
    print(f"Task has {len(history)} execution record(s)")

    # Get stats
    stats = await task_mgr.get_stats()
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"By status: {stats['by_status']}")
    print(f"Avg execution time: {stats['average_execution_time']:.2f}s")
```

### 7. Health Monitoring and Escalation

```python
async def monitoring_example():
    # Get health report for all agents
    health = await orchestrator.monitor_health()

    print(f"Total agents: {health['total_agents']}")
    for agent_id, status in health['agents'].items():
        print(f"  {status['name']}: {status['state']}")
        print(f"    - Uptime: {status['uptime_seconds']:.1f}s")
        print(f"    - Completed: {status['tasks_completed']}")
        print(f"    - Failed: {status['tasks_failed']}")

    # Escalate a task to human operator
    task = orchestrator.create_task(
        description="Ambiguous customer request",
        required_capabilities=["customer_support"]
    )

    success = await orchestrator.escalate_to_human(
        task_id=task.id,
        reason="Request requires human judgment",
        user_phone="+1234567890"
    )

    if success:
        print("Task escalated to WhatsApp")
```

### 8. Retry and Error Handling

```python
async def retry_example():
    # Create task
    task = orchestrator.create_task(
        description="Fetch external data",
        priority=TaskPriority.NORMAL,
        required_capabilities=["api_client"],
        max_retries=3,
        timeout_seconds=30
    )

    # Assign and try to execute
    if await orchestrator.route_task(task):
        result = await orchestrator.execute_task(task.id)

        # If failed and retries available
        current_task = orchestrator.get_task(task.id)
        if current_task.status == TaskStatus.FAILED and current_task.retry_count < 3:
            # Schedule retry with exponential backoff
            retry_success = await orchestrator.retry_task(task.id, delay_seconds=5.0)
            if retry_success:
                print("Task retry scheduled with backoff")
```

### 9. Configuration from .env File

Create `.env` file:

```env
# System
AGENTARMY_SYSTEM_SYSTEM_NAME=MyCode Horde
AGENTARMY_SYSTEM_ENVIRONMENT=production
AGENTARMY_SYSTEM_DEBUG=false
AGENTARMY_SYSTEM_LOG_LEVEL=INFO

# Redis
AGENTARMY_REDIS_HOST=redis.example.com
AGENTARMY_REDIS_PORT=6379
AGENTARMY_REDIS_DB=0
AGENTARMY_REDIS_PASSWORD=secret123

# RabbitMQ
AGENTARMY_RABBITMQ_HOST=rabbitmq.example.com
AGENTARMY_RABBITMQ_PORT=5672
AGENTARMY_RABBITMQ_USERNAME=admin
AGENTARMY_RABBITMQ_PASSWORD=secret456
AGENTARMY_RABBITMQ_VHOST=/

# Database
AGENTARMY_DATABASE_HOST=postgres.example.com
AGENTARMY_DATABASE_PORT=5432
AGENTARMY_DATABASE_NAME=codehorde
AGENTARMY_DATABASE_USER=dbuser
AGENTARMY_DATABASE_PASSWORD=dbpass

# Claude
AGENTARMY_CLAUDE_API_KEY=sk-xxx...
AGENTARMY_CLAUDE_DEFAULT_MODEL=claude-3-5-sonnet-20241022
AGENTARMY_CLAUDE_MAX_TOKENS=4096
AGENTARMY_CLAUDE_TIMEOUT=60.0

# Ollama
AGENTARMY_OLLAMA_BASE_URL=http://localhost:11434
AGENTARMY_OLLAMA_DEFAULT_MODEL=llama2

# WhatsApp
AGENTARMY_WHATSAPP_API_TOKEN=token_xxx...
AGENTARMY_WHATSAPP_PHONE_NUMBER_ID=1234567890
AGENTARMY_WHATSAPP_VERIFY_TOKEN=verify_xxx...
AGENTARMY_WHATSAPP_WEBHOOK_SECRET=webhook_xxx...

# Security
AGENTARMY_SECURITY_JWT_SECRET=your-secret-key-here
AGENTARMY_SECURITY_JWT_ALGORITHM=HS256
AGENTARMY_SECURITY_TOKEN_EXPIRY=3600
AGENTARMY_SECURITY_AUDIT_HASH_ALGORITHM=SHA256
```

### 10. Complete Example: Multi-Agent Data Pipeline

```python
import asyncio
from src.core.config import MasterSettings
from src.core.orchestrator import Orchestrator, TaskPriority
from src.core.agent_base import AgentCapability, AgentIdentity, BaseAgent
from typing import Any

class ExtractorAgent(BaseAgent):
    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        # Simulate data extraction
        return {"extracted_records": 1000}

class ValidatorAgent(BaseAgent):
    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        # Validate extracted data
        return {"valid_records": 950, "invalid_records": 50}

class TransformerAgent(BaseAgent):
    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        # Transform validated data
        return {"transformed_records": 950}

async def main():
    config = MasterSettings.from_env()
    orchestrator = Orchestrator()

    # Register agents
    agents = [
        ExtractorAgent(AgentIdentity(
            name="Extractor", role="extraction",
            capabilities=[AgentCapability(name="extract")]
        )),
        ValidatorAgent(AgentIdentity(
            name="Validator", role="validation",
            capabilities=[AgentCapability(name="validate")]
        )),
        TransformerAgent(AgentIdentity(
            name="Transformer", role="transformation",
            capabilities=[AgentCapability(name="transform")]
        ))
    ]

    for agent in agents:
        await orchestrator.register_agent(agent)

    # Create pipeline
    tasks = [
        orchestrator.create_task(
            "Extract data from source",
            TaskPriority.HIGH,
            required_capabilities=["extract"]
        ),
        orchestrator.create_task(
            "Validate extracted data",
            TaskPriority.NORMAL,
            required_capabilities=["validate"]
        ),
        orchestrator.create_task(
            "Transform data for output",
            TaskPriority.NORMAL,
            required_capabilities=["transform"]
        )
    ]

    # Execute sequential workflow
    results = await orchestrator.execute_workflow(
        "data_pipeline",
        "sequential",
        tasks
    )

    # Check results
    for task_id, result in results.items():
        print(f"Task {task_id}: {result}")

    # Health check
    health = await orchestrator.monitor_health()
    print(f"\nHealth Report: {health['total_agents']} agents active")

    # Cleanup
    for agent in agents:
        await orchestrator.unregister_agent(agent.identity.id)

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing with pytest-asyncio

```python
import pytest
from src.core.task_manager import TaskManager, TaskStatus

@pytest.mark.asyncio
async def test_create_task():
    mgr = TaskManager()
    task_id = await mgr.create_task("Test task")
    assert task_id in mgr.tasks

@pytest.mark.asyncio
async def test_task_lifecycle():
    mgr = TaskManager()
    task_id = await mgr.create_task("Lifecycle test")

    await mgr.assign_task(task_id, "agent_001")
    await mgr.update_task_status(task_id, TaskStatus.IN_PROGRESS)
    result = await mgr.complete_task(task_id, {"status": "success"})

    assert result.status == TaskStatus.COMPLETED
    assert result.execution_time_seconds > 0
```
