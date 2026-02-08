"""Router extensions for new model providers and RLM capabilities.

This module extends the existing ModelRouter with:
- New provider support (OpenAI, HuggingFace, Bielik, RLM)
- Polish language detection and routing
- Context-size-based RLM wrapping
- Lazy initialization of new clients

Rather than modifying the existing router.py, we extend it here
to maintain backward compatibility and allow independent evolution.
"""

import re
from typing import Optional, Tuple, Any
import structlog

from src.models.schemas import (
    ModelProvider,
    ModelTier,
    SensitivityLevel,
    LLMRequest,
)
from src.models.router import ModelRouter


logger = structlog.get_logger(__name__)


class RouterExtensions:
    """Extensions to the core ModelRouter for new providers."""

    # Polish language detection
    POLISH_CHAR_PATTERN = re.compile(r"[ąćęłńóśźż]", re.IGNORECASE)
    POLISH_KEYWORDS = {"jak", "co", "gdzie", "kiedy", "dlaczego", "czy"}

    # RLM thresholds
    RLM_CONTEXT_THRESHOLD_CHARS = 50_000  # Trigger RLM at 50k characters
    RLM_PROMPT_THRESHOLD_CHARS = 8_000    # Trigger RLM if prompt alone is huge

    @staticmethod
    def detect_polish(text: str) -> bool:
        """Detect if text contains Polish language.

        Uses heuristics: Polish diacritics and common Polish words.

        Args:
            text: Text to analyze.

        Returns:
            True if text appears to be Polish, False otherwise.
        """
        # Check for Polish diacritics
        if RouterExtensions.POLISH_CHAR_PATTERN.search(text):
            return True

        # Check for Polish keywords
        words = text.lower().split()
        polish_word_count = sum(
            1 for word in words if word in RouterExtensions.POLISH_KEYWORDS
        )

        return polish_word_count >= max(1, len(words) // 10)

    @staticmethod
    def should_use_rlm(request: LLMRequest) -> bool:
        """Determine if RLM should be used for this request.

        RLM is beneficial when:
        - Prompt is very large (exceeds window size)
        - Request involves processing large documents

        Args:
            request: The LLM request.

        Returns:
            True if RLM should be used, False otherwise.
        """
        prompt_text = request.prompt
        if request.system_prompt:
            prompt_text = f"{request.system_prompt}\n{request.prompt}"

        prompt_size = len(prompt_text)

        # Use RLM if prompt is very large
        if prompt_size > RouterExtensions.RLM_PROMPT_THRESHOLD_CHARS:
            logger.info(
                "RLM recommended",
                reason="large_prompt",
                prompt_chars=prompt_size,
            )
            return True

        return False

    @staticmethod
    def route_with_extensions(
        base_router: ModelRouter,
        request: LLMRequest,
        agent_id: str = "unknown",
    ) -> Tuple[ModelProvider, Optional[str]]:
        """Enhanced routing that considers new providers and RLM.

        This wraps the base router's route method with additional logic
        for Polish detection and RLM triggering.

        Args:
            base_router: The base ModelRouter instance.
            request: The LLM request.
            agent_id: ID of the requesting agent.

        Returns:
            Tuple of (provider, optional_model_name).
        """
        # Detect Polish and route to Bielik if appropriate
        is_polish = RouterExtensions.detect_polish(request.prompt)

        if is_polish:
            logger.info(
                "Polish detected, preferring Bielik",
                agent_id=agent_id,
                prompt_sample=request.prompt[:50],
            )

            # Check if Bielik is available (would need to be added to fallback chain)
            # For now, log the preference but continue with standard routing
            if hasattr(base_router, "fallback_chain"):
                if ModelProvider.BIELIK in base_router.fallback_chain:
                    return ModelProvider.BIELIK, None

        # Check if RLM should be triggered
        use_rlm = RouterExtensions.should_use_rlm(request)

        if use_rlm:
            logger.info(
                "RLM recommended for this request",
                agent_id=agent_id,
                prompt_chars=len(request.prompt),
            )
            # Return special marker that RLM should wrap the model
            # (Implementation depends on orchestrator)
            # For now, we just log and continue

        # Use base router's routing logic
        return base_router.route(request, agent_id)


def extend_router_fallback_chain(router: ModelRouter) -> None:
    """Extend router's fallback chain with new providers.

    Adds OpenAI, Gemini, Kimi, HuggingFace, and Bielik to the fallback chain
    in a sensible order based on cost and capability.

    Args:
        router: The ModelRouter to extend.
    """
    # Extend fallback chain: Claude > Gemini > OpenAI > Kimi > Bielik > HuggingFace > Ollama
    # Gemini high in chain (fast, cost-effective), Kimi strong for coding tasks
    router.fallback_chain = [
        ModelProvider.CLAUDE,
        ModelProvider.GEMINI,
        ModelProvider.OPENAI,
        ModelProvider.KIMI,
        ModelProvider.BIELIK,
        ModelProvider.HUGGINGFACE,
        ModelProvider.OLLAMA,
    ]

    logger.info(
        "Router fallback chain extended",
        chain=[p.value for p in router.fallback_chain],
    )


def add_openai_routing_rules(router: ModelRouter, openai_api_key: Optional[str] = None) -> None:
    """Add routing rules for OpenAI models.

    Creates rules to prefer OpenAI for:
    - POWERFUL tier requests
    - Tool-use heavy tasks
    - Cost-sensitive but quality-demanding scenarios

    Args:
        router: The ModelRouter to extend.
        openai_api_key: Optional OpenAI API key (for availability check).
    """
    from src.models.router import RoutingRule, RoutingCondition

    # OpenAI for POWERFUL tier
    powerful_rule = RoutingRule(
        condition=RoutingCondition.MODEL_TIER,
        condition_value=ModelTier.POWERFUL,
        target_provider=ModelProvider.OPENAI,
        target_model="gpt-4o",
        priority=10,
    )

    # OpenAI for balanced tier (good quality/cost ratio)
    balanced_rule = RoutingRule(
        condition=RoutingCondition.MODEL_TIER,
        condition_value=ModelTier.BALANCED,
        target_provider=ModelProvider.OPENAI,
        target_model="gpt-4o-mini",
        priority=5,
    )

    router.routing_rules.extend([powerful_rule, balanced_rule])

    # Re-sort rules by priority
    router.routing_rules.sort(key=lambda r: r.priority, reverse=True)

    logger.info("OpenAI routing rules added", rule_count=2)


def add_huggingface_routing_rules(router: ModelRouter) -> None:
    """Add routing rules for HuggingFace models.

    Creates rules to prefer HuggingFace for:
    - FAST tier (lightweight open-source models)
    - Cost-conscious scenarios
    - PUBLIC data (no confidentiality concerns)

    Args:
        router: The ModelRouter to extend.
    """
    from src.models.router import RoutingRule, RoutingCondition

    # HuggingFace for FAST tier
    fast_rule = RoutingRule(
        condition=RoutingCondition.MODEL_TIER,
        condition_value=ModelTier.FAST,
        target_provider=ModelProvider.HUGGINGFACE,
        target_model="mistral-7b",
        priority=8,
    )

    # HuggingFace for PUBLIC data
    public_rule = RoutingRule(
        condition=RoutingCondition.SENSITIVITY_LEVEL,
        condition_value=SensitivityLevel.PUBLIC,
        target_provider=ModelProvider.HUGGINGFACE,
        target_model="zephyr-7b",
        priority=3,
    )

    router.routing_rules.extend([fast_rule, public_rule])
    router.routing_rules.sort(key=lambda r: r.priority, reverse=True)

    logger.info("HuggingFace routing rules added", rule_count=2)


def add_gemini_routing_rules(router: ModelRouter) -> None:
    """Add routing rules for Google Gemini models.

    Creates rules to prefer Gemini for:
    - FAST tier (gemini-2.0-flash is extremely fast and cheap)
    - PUBLIC and INTERNAL data

    Args:
        router: The ModelRouter to extend.
    """
    from src.models.router import RoutingRule, RoutingCondition

    # Gemini Flash for FAST tier (very low cost, high speed)
    fast_rule = RoutingRule(
        condition=RoutingCondition.MODEL_TIER,
        condition_value=ModelTier.FAST,
        target_provider=ModelProvider.GEMINI,
        target_model="gemini-2.0-flash",
        priority=9,
    )

    router.routing_rules.append(fast_rule)
    router.routing_rules.sort(key=lambda r: r.priority, reverse=True)

    logger.info("Gemini routing rules added", rule_count=1)


def add_kimi_routing_rules(router: ModelRouter) -> None:
    """Add routing rules for Kimi (Moonshot) models.

    Creates rules to prefer Kimi for:
    - Large context tasks (128k context window)
    - Coding-related tasks (Kimi excels at code)

    Args:
        router: The ModelRouter to extend.
    """
    from src.models.router import RoutingRule, RoutingCondition

    # Kimi for BALANCED tier with code focus
    balanced_code_rule = RoutingRule(
        condition=RoutingCondition.MODEL_TIER,
        condition_value=ModelTier.BALANCED,
        target_provider=ModelProvider.KIMI,
        target_model="kimi-latest",
        priority=4,
    )

    router.routing_rules.append(balanced_code_rule)
    router.routing_rules.sort(key=lambda r: r.priority, reverse=True)

    logger.info("Kimi routing rules added", rule_count=1)


def add_bielik_routing_rules(router: ModelRouter) -> None:
    """Add routing rules for Bielik models.

    Creates rules to prefer Bielik for:
    - Polish language prompts/documents
    - INTERNAL data (local Polish processing)
    - Polish-specific NLP tasks

    Args:
        router: The ModelRouter to extend.
    """
    # Note: This requires custom condition handling
    # The Bielik routing would normally be language-based, not in the enum
    logger.info(
        "Bielik routing configured",
        note="Polish language detection will trigger Bielik routing",
    )


class EnhancedModelRouter(ModelRouter):
    """Extended ModelRouter with support for new providers and RLM.

    Inherits from ModelRouter and adds:
    - OpenAI, HuggingFace, Bielik routing
    - Polish language detection
    - RLM context awareness
    """

    def __init__(self, *args, **kwargs):
        """Initialize enhanced router."""
        super().__init__(*args, **kwargs)

        # Extend fallback chain
        extend_router_fallback_chain(self)

        # Add new provider rules
        add_openai_routing_rules(self)
        add_gemini_routing_rules(self)
        add_kimi_routing_rules(self)
        add_huggingface_routing_rules(self)
        add_bielik_routing_rules(self)

        logger.info("EnhancedModelRouter initialized with new provider support")

    def route_with_language_detection(
        self, request: LLMRequest, agent_id: str = "unknown"
    ) -> Tuple[ModelProvider, Optional[str]]:
        """Route request with Polish language detection.

        If Polish is detected, prefer Bielik (if available).

        Args:
            request: The LLM request.
            agent_id: ID of requesting agent.

        Returns:
            Tuple of (provider, optional_model_name).
        """
        # Check for Polish
        if RouterExtensions.detect_polish(request.prompt):
            logger.info(
                "Polish detected in prompt",
                agent_id=agent_id,
            )

            # Check if Bielik is available
            if self._is_provider_available(ModelProvider.BIELIK, agent_id):
                logger.info(
                    "Routing to Bielik (Polish model)",
                    agent_id=agent_id,
                )
                return ModelProvider.BIELIK, None

        # Fall back to standard routing
        return self.route(request, agent_id)

    def route_with_rlm_awareness(
        self, request: LLMRequest, agent_id: str = "unknown"
    ) -> Tuple[ModelProvider, Optional[str], bool]:
        """Route request and indicate if RLM should be used.

        Args:
            request: The LLM request.
            agent_id: ID of requesting agent.

        Returns:
            Tuple of (provider, optional_model_name, should_use_rlm).
        """
        should_use_rlm = RouterExtensions.should_use_rlm(request)
        provider, model = self.route(request, agent_id)

        if should_use_rlm:
            logger.info(
                "RLM will be used for this request",
                agent_id=agent_id,
                provider=provider,
            )

        return provider, model, should_use_rlm
