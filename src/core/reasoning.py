"""Chain-of-Thought (CoT) reasoning engine for AgentArmy multi-agent system.

This module provides structured reasoning capabilities with multiple strategies
including step-by-step reasoning, ReAct (Reason-Act-Observe), Tree of Thought,
and Self-Critique patterns.
"""

import re
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

from src.core.agent_base import AgentIdentity
from src.models.schemas import LLMRequest, LLMResponse, ModelTier, SensitivityLevel


logger = structlog.get_logger(__name__)


class ThoughtStep(BaseModel):
    """A single step in a reasoning chain.

    Represents one logical step in the agent's chain-of-thought process,
    including the reasoning content, confidence level, and resource usage.

    Attributes:
        step_number: Sequential step number (1-indexed).
        title: Brief title describing this step (e.g., "Understanding the task").
        content: The actual reasoning text for this step.
        confidence: Confidence level in this reasoning (0.0-1.0).
        duration_ms: Time spent on this step in milliseconds.
        tokens_used: Number of tokens consumed for this step.
    """

    step_number: int = Field(..., ge=1, description="Sequential step number")
    title: str = Field(..., min_length=1, description="Step title")
    content: str = Field(..., min_length=1, description="Reasoning content")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence level in this reasoning",
    )
    duration_ms: float = Field(default=0.0, ge=0.0, description="Step duration")
    tokens_used: int = Field(default=0, ge=0, description="Tokens consumed")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "step_number": 1,
                "title": "Understanding the task",
                "content": "The user wants me to...",
                "confidence": 0.95,
                "duration_ms": 125.5,
                "tokens_used": 142,
            }
        }


class ReasoningChain(BaseModel):
    """A complete chain of reasoning with multiple steps and conclusion.

    Encapsulates the full reasoning process from an agent, including all
    intermediate steps, final conclusion, resource usage, and metadata.

    Attributes:
        task_id: Unique identifier for the task being reasoned about.
        agent_id: ID of the agent performing the reasoning.
        agent_role: Role/type of the agent.
        steps: Sequence of reasoning steps.
        conclusion: Final synthesized answer or result.
        total_tokens: Total tokens used across all steps.
        total_duration_ms: Total time spent reasoning.
        model_used: Name of the LLM model used.
        provider: LLM provider (claude, openai, etc.).
        timestamp: When the reasoning was generated.
    """

    task_id: str = Field(..., description="Unique task identifier")
    agent_id: str = Field(..., description="Agent performing reasoning")
    agent_role: str = Field(..., description="Agent's role/type")
    steps: list[ThoughtStep] = Field(
        default_factory=list, description="Sequence of reasoning steps"
    )
    conclusion: str = Field(..., description="Final synthesized answer")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens used")
    total_duration_ms: float = Field(default=0.0, ge=0.0, description="Total duration")
    model_used: str = Field(..., description="LLM model name")
    provider: str = Field(..., description="LLM provider")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Generation timestamp",
    )

    def summary(self) -> str:
        """Generate a formatted string representation of the reasoning chain.

        Returns:
            A human-readable summary of the entire reasoning process.
        """
        lines = [
            f"Reasoning Chain for Task: {self.task_id}",
            f"Agent: {self.agent_role} ({self.agent_id})",
            f"Model: {self.model_used} ({self.provider})",
            f"Generated: {self.timestamp.isoformat()}",
            f"Stats: {len(self.steps)} steps, {self.total_tokens} tokens, "
            f"{self.total_duration_ms:.1f}ms",
            "",
            "=== REASONING STEPS ===",
        ]

        for step in self.steps:
            lines.extend(
                [
                    f"\nStep {step.step_number}: {step.title}",
                    f"Confidence: {step.confidence:.1%} | Duration: {step.duration_ms:.1f}ms | "
                    f"Tokens: {step.tokens_used}",
                    "-" * 60,
                    step.content,
                ]
            )

        lines.extend(
            [
                "",
                "=== CONCLUSION ===",
                self.conclusion,
            ]
        )

        return "\n".join(lines)

    def to_audit_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for audit logging.

        Returns:
            Dictionary representation suitable for audit trail storage.
        """
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "step_count": len(self.steps),
            "steps": [
                {
                    "number": step.step_number,
                    "title": step.title,
                    "confidence": step.confidence,
                    "duration_ms": step.duration_ms,
                    "tokens": step.tokens_used,
                }
                for step in self.steps
            ],
            "conclusion_length": len(self.conclusion),
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "model_used": self.model_used,
            "provider": self.provider,
            "timestamp": self.timestamp.isoformat(),
        }


class ReasoningStrategy(str, Enum):
    """Strategies for chain-of-thought reasoning.

    Attributes:
        STEP_BY_STEP: Classic CoT - decompose and reason through each step.
        REACT: Reason-Act-Observe cycle for interactive tasks.
        TREE_OF_THOUGHT: Explore multiple reasoning branches, select best.
        SELF_CRITIQUE: Generate answer, critique, and refine iteratively.
    """

    STEP_BY_STEP = "step_by_step"
    REACT = "react"
    TREE_OF_THOUGHT = "tree_of_thought"
    SELF_CRITIQUE = "self_critique"


class ChainOfThoughtEngine:
    """Engine for generating structured chain-of-thought reasoning.

    Provides methods to build LLM requests with CoT prompting and parse
    responses into structured reasoning chains. Supports multiple reasoning
    strategies and integrates with the LLM router infrastructure.

    The reasoning process is split into two phases:
    1. build_reasoning_request: Creates an LLMRequest with strategy-specific prompt
    2. parse_reasoning_response: Parses LLMResponse into a ReasoningChain

    This design allows the actual LLM invocation to occur in the agent layer,
    which has access to the router and appropriate credentials.
    """

    def __init__(self) -> None:
        """Initialize the Chain-of-Thought engine."""
        self._logger = structlog.get_logger("cot_engine")

    def build_reasoning_request(
        self,
        task: dict[str, Any],
        agent_identity: AgentIdentity,
        system_context: str,
        strategy: ReasoningStrategy = ReasoningStrategy.STEP_BY_STEP,
        model_tier: ModelTier = ModelTier.BALANCED,
        sensitivity: SensitivityLevel = SensitivityLevel.INTERNAL,
        max_tokens: int = 4096,
    ) -> LLMRequest:
        """Build an LLMRequest for chain-of-thought reasoning.

        Constructs a request with strategy-specific prompts that guide the LLM
        through structured reasoning. The resulting request can be passed to
        the LLM router for execution.

        Args:
            task: Task specification dictionary with at least 'id' and 'description'.
            agent_identity: Identity and metadata of the reasoning agent.
            system_context: Role-specific system prompt/context.
            strategy: Reasoning strategy to use (default: STEP_BY_STEP).
            model_tier: Preferred model tier for this request.
            sensitivity: Data sensitivity level for routing decisions.
            max_tokens: Maximum tokens in the response.

        Returns:
            LLMRequest ready to be sent to an LLM provider.

        Raises:
            ValueError: If task is missing required fields.
        """
        if "id" not in task or "description" not in task:
            raise ValueError("Task must contain 'id' and 'description' fields")

        system_prompt, user_prompt = self._build_cot_prompt(
            task=task,
            agent_role=agent_identity.role,
            agent_name=agent_identity.name,
            system_context=system_context,
            strategy=strategy,
        )

        self._logger.info(
            "building_cot_request",
            task_id=task.get("id"),
            agent_id=agent_identity.id,
            strategy=strategy.value,
        )

        return LLMRequest(
            system_prompt=system_prompt,
            prompt=user_prompt,
            model_preference=model_tier,
            sensitivity=sensitivity,
            max_tokens=max_tokens,
            temperature=0.5,  # Lower temperature for more deterministic reasoning
        )

    def parse_reasoning_response(
        self,
        response: LLMResponse,
        task: dict[str, Any],
        agent_identity: AgentIdentity,
        strategy: ReasoningStrategy = ReasoningStrategy.STEP_BY_STEP,
    ) -> ReasoningChain:
        """Parse an LLM response into a structured reasoning chain.

        Extracts reasoning steps and conclusion from the LLM's response using
        strategy-specific parsing. Handles various formatting variations.

        Args:
            response: LLMResponse from an LLM provider.
            task: Original task specification.
            agent_identity: Identity of the reasoning agent.
            strategy: Reasoning strategy used (for parsing guidance).

        Returns:
            ReasoningChain with structured steps and conclusion.

        Raises:
            ValueError: If response cannot be parsed into valid structure.
        """
        self._logger.info(
            "parsing_cot_response",
            task_id=task.get("id"),
            agent_id=agent_identity.id,
            response_length=len(response.content),
        )

        steps, conclusion = self._parse_reasoning_response(
            raw_response=response.content, strategy=strategy
        )

        # Validate we got at least a conclusion
        if not conclusion or not conclusion.strip():
            self._logger.warning(
                "no_conclusion_found",
                task_id=task.get("id"),
                response_length=len(response.content),
            )
            conclusion = response.content  # Fallback to raw response

        # Calculate totals
        total_tokens = sum(step.tokens_used for step in steps)
        total_tokens += response.tokens_used or 0

        reasoning_chain = ReasoningChain(
            task_id=task.get("id", "unknown"),
            agent_id=agent_identity.id,
            agent_role=agent_identity.role,
            steps=steps,
            conclusion=conclusion,
            total_tokens=total_tokens,
            total_duration_ms=response.latency_ms or 0.0,
            model_used=response.model_used,
            provider=response.provider.value,
        )

        self._logger.info(
            "cot_parsing_complete",
            task_id=task.get("id"),
            step_count=len(steps),
            total_tokens=total_tokens,
        )

        return reasoning_chain

    def _build_cot_prompt(
        self,
        task: dict[str, Any],
        agent_role: str,
        agent_name: str,
        system_context: str,
        strategy: ReasoningStrategy,
    ) -> tuple[str, str]:
        """Build system and user prompts for chain-of-thought reasoning.

        Routes to strategy-specific prompt builders.

        Args:
            task: Task specification.
            agent_role: Agent's role.
            agent_name: Agent's name.
            system_context: Role-specific context.
            strategy: Reasoning strategy.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        if strategy == ReasoningStrategy.STEP_BY_STEP:
            system_prompt = self._build_step_by_step_system_prompt(
                agent_name=agent_name,
                agent_role=agent_role,
                system_context=system_context,
            )
            user_prompt = self._build_step_by_step_user_prompt(task)

        elif strategy == ReasoningStrategy.REACT:
            system_prompt = self._build_react_system_prompt(
                agent_name=agent_name,
                agent_role=agent_role,
                system_context=system_context,
            )
            user_prompt = self._build_react_user_prompt(task)

        elif strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            system_prompt = self._build_tree_of_thought_system_prompt(
                agent_name=agent_name,
                agent_role=agent_role,
                system_context=system_context,
            )
            user_prompt = self._build_tree_of_thought_user_prompt(task)

        elif strategy == ReasoningStrategy.SELF_CRITIQUE:
            system_prompt = self._build_self_critique_system_prompt(
                agent_name=agent_name,
                agent_role=agent_role,
                system_context=system_context,
            )
            user_prompt = self._build_self_critique_user_prompt(task)

        else:
            raise ValueError(f"Unknown reasoning strategy: {strategy}")

        return system_prompt, user_prompt

    def _build_step_by_step_system_prompt(
        self, agent_name: str, agent_role: str, system_context: str
    ) -> str:
        """Build system prompt for STEP_BY_STEP reasoning.

        Args:
            agent_name: Name of the agent.
            agent_role: Role of the agent.
            system_context: Additional context for the agent's role.

        Returns:
            System prompt string.
        """
        return f"""You are {agent_name}, a {agent_role} in the AgentArmy system.
{system_context}

When solving tasks, think step by step. Structure your response with the following format:

<step title="Understanding the task">
[Your analysis of what needs to be done and the key requirements]
</step>

<step title="Planning the approach">
[Your strategy for solving this, including any assumptions or constraints]
</step>

<step title="Executing the solution">
[Your detailed work, calculations, analysis, or findings]
</step>

<step title="Validating results">
[Your verification that the output is correct and complete]
</step>

<conclusion>
[Your final answer/result synthesizing all steps clearly and concisely]
</conclusion>

Be thorough in each step but concise in your language. Clearly separate your thinking process from your final answer."""

    def _build_step_by_step_user_prompt(self, task: dict[str, Any]) -> str:
        """Build user prompt for STEP_BY_STEP reasoning.

        Args:
            task: Task specification.

        Returns:
            User prompt string.
        """
        task_id = task.get("id", "unknown")
        description = task.get("description", "")
        context = task.get("context", "")
        constraints = task.get("constraints", "")

        parts = [f"Task ID: {task_id}", f"\nTask Description:\n{description}"]

        if context:
            parts.append(f"\nAdditional Context:\n{context}")

        if constraints:
            parts.append(f"\nConstraints:\n{constraints}")

        parts.append(
            "\nPlease reason through this task step by step, using the format "
            "specified in the system prompt."
        )

        return "\n".join(parts)

    def _build_react_system_prompt(
        self, agent_name: str, agent_role: str, system_context: str
    ) -> str:
        """Build system prompt for REACT (Reason-Act-Observe) reasoning.

        Args:
            agent_name: Name of the agent.
            agent_role: Role of the agent.
            system_context: Additional context for the agent's role.

        Returns:
            System prompt string.
        """
        return f"""You are {agent_name}, a {agent_role} in the AgentArmy system.
{system_context}

Use the ReAct (Reason-Act-Observe) framework to solve tasks iteratively:

1. Thought: Analyze the current state and think about what to do next
2. Action: Describe what action you would take
3. Observation: Describe what you observe from taking that action

Repeat this cycle as needed. Structure your response as:

<step title="Thought">
[Your analysis of the current state and what you need to do]
</step>

<step title="Action">
[The specific action you would take]
</step>

<step title="Observation">
[What you observe from this action; this may trigger another cycle]
</step>

[Repeat Thought-Action-Observation cycles as needed]

<conclusion>
[Your final result once the task is complete]
</conclusion>

Focus on concrete, observable actions and their outcomes."""

    def _build_react_user_prompt(self, task: dict[str, Any]) -> str:
        """Build user prompt for REACT reasoning.

        Args:
            task: Task specification.

        Returns:
            User prompt string.
        """
        task_id = task.get("id", "unknown")
        description = task.get("description", "")

        return (
            f"Task ID: {task_id}\n\n"
            f"Task:\n{description}\n\n"
            f"Please reason through this task using the ReAct framework, "
            f"iterating with Thought-Action-Observation cycles as needed."
        )

    def _build_tree_of_thought_system_prompt(
        self, agent_name: str, agent_role: str, system_context: str
    ) -> str:
        """Build system prompt for TREE_OF_THOUGHT reasoning.

        Args:
            agent_name: Name of the agent.
            agent_role: Role of the agent.
            system_context: Additional context for the agent's role.

        Returns:
            System prompt string.
        """
        return f"""You are {agent_name}, a {agent_role} in the AgentArmy system.
{system_context}

Use Tree of Thought reasoning: explore multiple distinct approaches to solving
the problem, then evaluate and select the best one.

Structure your response as:

<step title="Branch A: [approach name]">
[Detailed reasoning for first approach]
</step>

<step title="Branch B: [approach name]">
[Detailed reasoning for alternative approach]
</step>

<step title="Branch C: [approach name]">
[Detailed reasoning for another approach (optional)]
</step>

<step title="Evaluation">
[Compare the branches, analyze strengths/weaknesses, select the best approach]
</step>

<conclusion>
[Final result based on the selected best approach]
</conclusion>

Be creative in exploring diverse solution paths. Focus the evaluation on
feasibility, correctness, and efficiency."""

    def _build_tree_of_thought_user_prompt(self, task: dict[str, Any]) -> str:
        """Build user prompt for TREE_OF_THOUGHT reasoning.

        Args:
            task: Task specification.

        Returns:
            User prompt string.
        """
        task_id = task.get("id", "unknown")
        description = task.get("description", "")

        return (
            f"Task ID: {task_id}\n\n"
            f"Task:\n{description}\n\n"
            f"Please solve this using Tree of Thought reasoning. Explore at least "
            f"2-3 different approaches, then compare them and select the best one."
        )

    def _build_self_critique_system_prompt(
        self, agent_name: str, agent_role: str, system_context: str
    ) -> str:
        """Build system prompt for SELF_CRITIQUE reasoning.

        Args:
            agent_name: Name of the agent.
            agent_role: Role of the agent.
            system_context: Additional context for the agent's role.

        Returns:
            System prompt string.
        """
        return f"""You are {agent_name}, a {agent_role} in the AgentArmy system.
{system_context}

Use Self-Critique reasoning: first generate an initial solution, then critically
review it for errors or improvements, then provide a refined solution.

Structure your response as:

<step title="Initial solution">
[Your first attempt at solving the task]
</step>

<step title="Critical review">
[Carefully critique your initial solution. What could be wrong? What could be improved?
What edge cases did you miss? Are there logical errors?]
</step>

<step title="Refined solution">
[Improved solution that addresses the issues and critiques you identified]
</step>

<conclusion>
[Final polished and verified result]
</conclusion>

Be rigorous in your critique and make substantive improvements in the refined solution."""

    def _build_self_critique_user_prompt(self, task: dict[str, Any]) -> str:
        """Build user prompt for SELF_CRITIQUE reasoning.

        Args:
            task: Task specification.

        Returns:
            User prompt string.
        """
        task_id = task.get("id", "unknown")
        description = task.get("description", "")

        return (
            f"Task ID: {task_id}\n\n"
            f"Task:\n{description}\n\n"
            f"Please solve this using self-critique reasoning. First give an answer, "
            f"then critically review it, then provide an improved answer."
        )

    def _parse_reasoning_response(
        self, raw_response: str, strategy: ReasoningStrategy
    ) -> tuple[list[ThoughtStep], str]:
        """Parse an LLM response into reasoning steps and conclusion.

        Uses regex to extract <step> and <conclusion> blocks. Handles various
        formatting variations and missing tags gracefully.

        Args:
            raw_response: Raw text response from the LLM.
            strategy: Reasoning strategy (for context).

        Returns:
            Tuple of (list of ThoughtStep objects, conclusion string).
        """
        steps: list[ThoughtStep] = []
        conclusion = ""

        # Extract conclusion first
        conclusion_match = re.search(
            r"<conclusion>\s*(.*?)\s*</conclusion>",
            raw_response,
            re.DOTALL | re.IGNORECASE,
        )
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
        else:
            # Fallback: last paragraph might be conclusion
            paragraphs = [p.strip() for p in raw_response.split("\n\n") if p.strip()]
            if paragraphs:
                conclusion = paragraphs[-1]

        # Extract steps
        step_pattern = r'<step\s+title="([^"]+)"\s*>\s*(.*?)\s*</step>'
        step_matches = re.finditer(
            step_pattern, raw_response, re.DOTALL | re.IGNORECASE
        )

        step_number = 1
        for match in step_matches:
            title = match.group(1).strip()
            content = match.group(2).strip()

            if title and content:
                step = ThoughtStep(
                    step_number=step_number,
                    title=title,
                    content=content,
                    confidence=0.8,  # Default confidence
                    duration_ms=0.0,  # Would be set by actual execution
                    tokens_used=0,  # Would be calculated from token count
                )
                steps.append(step)
                step_number += 1

        # If no steps found, try to extract paragraphs as steps
        if not steps:
            self._logger.warning(
                "no_structured_steps_found",
                response_length=len(raw_response),
                attempting_fallback=True,
            )
            paragraphs = [
                p.strip() for p in raw_response.split("\n\n") if p.strip()
            ]
            for i, para in enumerate(paragraphs[:-1]):  # Exclude last para (conclusion)
                steps.append(
                    ThoughtStep(
                        step_number=i + 1,
                        title=f"Step {i + 1}",
                        content=para,
                        confidence=0.7,
                        duration_ms=0.0,
                        tokens_used=0,
                    )
                )

        return steps, conclusion
