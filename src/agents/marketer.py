"""LinkedIn content marketing agent with post writing and image generation."""

import json
from datetime import datetime, timezone
from typing import Any

import structlog

from src.core.agent_base import AgentCapability, AgentIdentity, AgentState, BaseAgent

logger = structlog.get_logger(__name__)


class MarketerAgent(BaseAgent):
    """Content marketing agent specializing in LinkedIn posts with AI-generated images.

    Responsibilities:
    - Writes engaging LinkedIn posts (hooks, body, CTAs, hashtags)
    - Generates relevant images for posts via AI image generation (DALL-E / Gemini)
    - Creates complete LinkedIn content packages (post + image)
    - Repurposes existing content (articles, blog posts) into LinkedIn format
    - A/B tests different post variations

    Capabilities:
    - write_linkedin_post: Write an optimized LinkedIn post
    - generate_post_image: Generate an AI image for a LinkedIn post
    - create_content_package: Full LinkedIn content package (post + image prompt)
    - repurpose_content: Turn existing content into LinkedIn posts
    """

    # LinkedIn post best-practice constraints
    MAX_POST_LENGTH = 3000
    OPTIMAL_POST_LENGTH = 1300  # Sweet spot for engagement
    MAX_HASHTAGS = 5

    def __init__(
        self,
        agent_id: str = "marketer-linkedin",
        name: str = "Marketer LinkedIn Agent",
        role: str = "marketer",
    ) -> None:
        """Initialize the LinkedIn Marketer agent.

        Args:
            agent_id: Unique agent identifier.
            name: Display name for the agent.
            role: Agent role classification.
        """
        identity = AgentIdentity(
            id=agent_id,
            name=name,
            role=role,
            security_level=2,
            capabilities=[
                AgentCapability(
                    name="write_linkedin_post",
                    version="1.0.0",
                    description="Write an optimized LinkedIn post with hook, body, CTA and hashtags",
                    parameters={
                        "topic": "str",
                        "tone": "str",
                        "target_audience": "str",
                        "include_cta": "bool",
                        "post_style": "str",
                    },
                ),
                AgentCapability(
                    name="generate_post_image",
                    version="1.0.0",
                    description="Generate an AI image prompt and metadata for a LinkedIn post",
                    parameters={
                        "post_content": "str",
                        "style": "str",
                        "aspect_ratio": "str",
                    },
                ),
                AgentCapability(
                    name="create_content_package",
                    version="1.0.0",
                    description="Create complete LinkedIn content package: post + image",
                    parameters={
                        "topic": "str",
                        "tone": "str",
                        "target_audience": "str",
                        "brand_voice": "str",
                    },
                ),
                AgentCapability(
                    name="repurpose_content",
                    version="1.0.0",
                    description="Transform existing content (articles, blogs) into LinkedIn posts",
                    parameters={
                        "source_content": "str",
                        "source_type": "str",
                        "num_variations": "int",
                    },
                ),
            ],
        )
        super().__init__(identity)
        self._posts_created: list[dict[str, Any]] = []
        self._images_generated: list[dict[str, Any]] = []

    async def startup(self) -> None:
        """Initialize marketer agent.

        Raises:
            Exception: If startup fails.
        """
        await super().startup()
        await logger.ainfo(
            "marketer_startup",
            agent_id=self.identity.id,
        )

    async def shutdown(self) -> None:
        """Shutdown marketer agent gracefully.

        Raises:
            Exception: If shutdown fails.
        """
        await logger.ainfo(
            "marketer_shutdown",
            agent_id=self.identity.id,
            posts_created=len(self._posts_created),
            images_generated=len(self._images_generated),
        )
        await super().shutdown()

    def _get_system_context(self) -> str:
        """Get system context for LLM reasoning.

        Returns:
            System context string describing the agent's role and expertise.
        """
        return (
            "You are a senior LinkedIn content strategist and copywriter. "
            "You specialize in writing viral, high-engagement LinkedIn posts "
            "that combine storytelling, professional insights, and strong hooks. "
            "You understand LinkedIn's algorithm: posts that get early engagement "
            "(comments > reactions > shares) get boosted. You know that "
            "short-form posts (under 1300 chars) with a compelling hook in "
            "the first 2 lines perform best. You create image prompts that "
            "are visually striking, professional, and complement the post message. "
            "You're an expert at writing for different tones: thought leadership, "
            "personal story, educational, contrarian, and celebratory."
        )

    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process LinkedIn marketing tasks.

        Supported task types:
        - write_linkedin_post: Write a LinkedIn post
        - generate_post_image: Generate image for a post
        - create_content_package: Full post + image package
        - repurpose_content: Repurpose existing content

        Args:
            task: Task payload with type and parameters.

        Returns:
            Result dictionary with generated content.

        Raises:
            ValueError: If task type is unsupported.
        """
        task_type = task.get("type", "unknown")
        task_id = task.get("id", "unknown")

        await logger.ainfo(
            "marketer_processing_task",
            task_id=task_id,
            task_type=task_type,
        )

        # ── Chain-of-Thought reasoning ────────────────────────────
        reasoning = await self.think(task)
        task.setdefault("context", {})["_reasoning"] = reasoning.conclusion

        try:
            if task_type == "write_linkedin_post":
                result = await self._handle_write_post(task)
            elif task_type == "generate_post_image":
                result = await self._handle_generate_image(task)
            elif task_type == "create_content_package":
                result = await self._handle_content_package(task)
            elif task_type == "repurpose_content":
                result = await self._handle_repurpose_content(task)
            else:
                return await self._handle_chat_message(task)
            result["reasoning"] = reasoning.to_audit_dict()
            return result
        except Exception as exc:
            await logger.aerror(
                "marketer_task_error",
                task_id=task_id,
                error=str(exc),
            )
            raise

    # ── Task Handlers ─────────────────────────────────────────────

    async def _handle_write_post(self, task: dict[str, Any]) -> dict[str, Any]:
        """Write a LinkedIn post.

        Uses LLM to generate an engaging LinkedIn post with proper structure:
        hook → body → call-to-action → hashtags.

        Args:
            task: Task with post parameters.

        Returns:
            Dictionary with the generated post and metadata.
        """
        ctx = task.get("context", {})
        topic = ctx.get("topic", "AI and the future of work")
        tone = ctx.get("tone", "thought_leadership")
        target_audience = ctx.get("target_audience", "tech professionals")
        include_cta = ctx.get("include_cta", True)
        post_style = ctx.get("post_style", "storytelling")

        await logger.ainfo(
            "writing_linkedin_post",
            topic=topic,
            tone=tone,
            style=post_style,
        )

        # Build the LLM prompt for post generation
        reasoning_context = ctx.get("_reasoning", "")

        post = await self._generate_post_with_llm(
            topic=topic,
            tone=tone,
            target_audience=target_audience,
            include_cta=include_cta,
            post_style=post_style,
            reasoning_context=reasoning_context,
        )

        self._posts_created.append(post)

        result = {
            "status": "completed",
            "post": post,
            "metadata": {
                "topic": topic,
                "tone": tone,
                "target_audience": target_audience,
                "style": post_style,
                "char_count": len(post.get("content", "")),
                "hashtag_count": len(post.get("hashtags", [])),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await logger.ainfo(
            "linkedin_post_created",
            char_count=len(post.get("content", "")),
        )

        return result

    async def _handle_generate_image(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate an AI image prompt for a LinkedIn post.

        Creates a detailed image generation prompt and metadata that can
        be sent to DALL-E, Midjourney, or Gemini's image generation.

        Args:
            task: Task with image generation parameters.

        Returns:
            Dictionary with image prompt and metadata.
        """
        ctx = task.get("context", {})
        post_content = ctx.get("post_content", "")
        style = ctx.get("style", "professional_modern")
        aspect_ratio = ctx.get("aspect_ratio", "1200x627")  # LinkedIn recommended

        await logger.ainfo(
            "generating_post_image",
            style=style,
            aspect_ratio=aspect_ratio,
        )

        image_spec = await self._generate_image_spec(
            post_content=post_content,
            style=style,
            aspect_ratio=aspect_ratio,
        )

        self._images_generated.append(image_spec)

        result = {
            "status": "completed",
            "image_spec": image_spec,
            "metadata": {
                "style": style,
                "aspect_ratio": aspect_ratio,
                "generation_ready": True,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await logger.ainfo(
            "post_image_spec_created",
            prompt_length=len(image_spec.get("prompt", "")),
        )

        return result

    async def _handle_content_package(self, task: dict[str, Any]) -> dict[str, Any]:
        """Create a complete LinkedIn content package (post + image).

        Generates both the post text and a matching image prompt as a
        cohesive content unit.

        Args:
            task: Task with content package parameters.

        Returns:
            Dictionary with post and image specification.
        """
        ctx = task.get("context", {})
        topic = ctx.get("topic", "AI and the future of work")
        tone = ctx.get("tone", "thought_leadership")
        target_audience = ctx.get("target_audience", "tech professionals")
        brand_voice = ctx.get("brand_voice", "professional but approachable")

        await logger.ainfo(
            "creating_content_package",
            topic=topic,
            tone=tone,
        )

        reasoning_context = ctx.get("_reasoning", "")

        # Step 1: Generate the post
        post = await self._generate_post_with_llm(
            topic=topic,
            tone=tone,
            target_audience=target_audience,
            include_cta=True,
            post_style="storytelling",
            reasoning_context=reasoning_context,
        )

        # Step 2: Generate matching image spec
        image_spec = await self._generate_image_spec(
            post_content=post.get("content", ""),
            style="professional_modern",
            aspect_ratio="1200x627",
        )

        self._posts_created.append(post)
        self._images_generated.append(image_spec)

        result = {
            "status": "completed",
            "content_package": {
                "post": post,
                "image": image_spec,
                "brand_voice": brand_voice,
            },
            "metadata": {
                "topic": topic,
                "tone": tone,
                "target_audience": target_audience,
                "post_char_count": len(post.get("content", "")),
                "has_image": True,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await logger.ainfo(
            "content_package_created",
            post_chars=len(post.get("content", "")),
        )

        return result

    async def _handle_repurpose_content(self, task: dict[str, Any]) -> dict[str, Any]:
        """Repurpose existing content into LinkedIn posts.

        Takes a blog post, article, or other content and creates
        one or more LinkedIn post variations.

        Args:
            task: Task with source content and parameters.

        Returns:
            Dictionary with repurposed LinkedIn post variations.
        """
        ctx = task.get("context", {})
        source_content = ctx.get("source_content", "")
        source_type = ctx.get("source_type", "blog_post")
        num_variations = min(ctx.get("num_variations", 3), 5)

        await logger.ainfo(
            "repurposing_content",
            source_type=source_type,
            source_length=len(source_content),
            variations_requested=num_variations,
        )

        # Generate variations using LLM reasoning
        reasoning_context = ctx.get("_reasoning", "")

        variations = []
        post_styles = ["storytelling", "listicle", "contrarian", "personal_insight", "educational"]

        for i in range(num_variations):
            style = post_styles[i % len(post_styles)]
            post = await self._generate_post_with_llm(
                topic=f"Repurpose from {source_type}: {source_content[:200]}",
                tone="thought_leadership",
                target_audience="professionals",
                include_cta=True,
                post_style=style,
                reasoning_context=reasoning_context,
            )
            post["variation_index"] = i + 1
            post["style"] = style
            variations.append(post)

        self._posts_created.extend(variations)

        result = {
            "status": "completed",
            "variations": variations,
            "metadata": {
                "source_type": source_type,
                "source_length": len(source_content),
                "variations_created": len(variations),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        await logger.ainfo(
            "content_repurposed",
            variations_created=len(variations),
        )

        return result

    # ── LLM Integration ───────────────────────────────────────────

    async def _generate_post_with_llm(
        self,
        topic: str,
        tone: str,
        target_audience: str,
        include_cta: bool,
        post_style: str,
        reasoning_context: str = "",
    ) -> dict[str, Any]:
        """Generate a LinkedIn post using the LLM.

        Constructs a detailed prompt and calls the LLM to generate
        an optimized LinkedIn post.

        Args:
            topic: Post topic/subject.
            tone: Writing tone (thought_leadership, personal, educational, etc.).
            target_audience: Who the post is aimed at.
            include_cta: Whether to include a call-to-action.
            post_style: Post format (storytelling, listicle, contrarian, etc.).
            reasoning_context: Pre-computed reasoning context from CoT engine.

        Returns:
            Dictionary with post content and components.
        """
        from src.models.schemas import LLMRequest, LLMResponse, ModelProvider

        prompt = (
            f"Write a high-engagement LinkedIn post about: {topic}\n\n"
            f"REQUIREMENTS:\n"
            f"- Tone: {tone}\n"
            f"- Target audience: {target_audience}\n"
            f"- Style: {post_style}\n"
            f"- Max length: {self.OPTIMAL_POST_LENGTH} characters\n"
            f"- Include CTA: {'yes' if include_cta else 'no'}\n"
            f"- Max hashtags: {self.MAX_HASHTAGS}\n\n"
            f"STRUCTURE (respond in this exact JSON format):\n"
            f'{{\n'
            f'  "hook": "First 2 lines that grab attention (show in preview)",\n'
            f'  "body": "Main content with insights, story, or value",\n'
            f'  "cta": "Call to action (if requested)",\n'
            f'  "hashtags": ["hashtag1", "hashtag2", "hashtag3"],\n'
            f'  "content": "The complete post text ready to publish"\n'
            f'}}\n\n'
            f"LINKEDIN BEST PRACTICES:\n"
            f"- Hook must create curiosity or make a bold claim\n"
            f"- Use line breaks for readability\n"
            f"- Include a personal angle or specific data point\n"
            f"- End with engagement driver (question or opinion request)\n"
        )

        if reasoning_context:
            prompt += f"\nPRE-ANALYSIS:\n{reasoning_context}\n"

        system_prompt = self._get_system_context()

        try:
            from src.models.claude_client import ClaudeClient

            client = ClaudeClient()
            llm_request = LLMRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2048,
                temperature=0.8,  # Slightly creative
            )
            response = await client.complete(llm_request)

            # Try to parse JSON from response
            content = response.content.strip()
            # Extract JSON if wrapped in markdown code block
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            try:
                post_data = json.loads(content)
                return post_data
            except json.JSONDecodeError:
                # If JSON parsing fails, treat the whole response as the post
                return {
                    "hook": content[:150],
                    "body": content,
                    "cta": "",
                    "hashtags": [],
                    "content": content,
                }

        except Exception as llm_exc:
            await logger.awarning(
                "post_llm_fallback",
                error=str(llm_exc),
            )
            # Fallback: return a template-based post
            return self._generate_fallback_post(
                topic, tone, target_audience, include_cta, post_style,
            )

    async def _generate_image_spec(
        self,
        post_content: str,
        style: str = "professional_modern",
        aspect_ratio: str = "1200x627",
    ) -> dict[str, Any]:
        """Generate an image specification for a LinkedIn post.

        Creates a detailed prompt for image generation APIs (DALL-E, Gemini)
        plus metadata for rendering.

        Args:
            post_content: The LinkedIn post text to create an image for.
            style: Visual style for the image.
            aspect_ratio: Image dimensions.

        Returns:
            Dictionary with image generation prompt and metadata.
        """
        from src.models.schemas import LLMRequest, ModelProvider

        # Style mappings for image generation
        style_descriptors = {
            "professional_modern": "clean, modern, professional, corporate, minimal design, blue and white tones",
            "tech_futuristic": "futuristic, digital, neon accents, tech-inspired, dark background",
            "warm_personal": "warm colors, friendly, inviting, soft lighting, personal feel",
            "data_driven": "infographic style, charts, data visualization, clean typography",
            "minimalist": "minimalist, lots of white space, single focal point, elegant",
        }

        style_desc = style_descriptors.get(style, style_descriptors["professional_modern"])

        # Generate image prompt using LLM
        prompt = (
            f"Based on this LinkedIn post, create a DALL-E image generation prompt:\n\n"
            f"POST:\n{post_content[:500]}\n\n"
            f"VISUAL STYLE: {style_desc}\n"
            f"ASPECT RATIO: {aspect_ratio}\n\n"
            f"Respond in this exact JSON format:\n"
            f'{{\n'
            f'  "prompt": "Detailed image generation prompt (max 400 chars)",\n'
            f'  "negative_prompt": "Things to avoid in the image",\n'
            f'  "alt_text": "Accessibility alt text for the image",\n'
            f'  "color_palette": ["#hex1", "#hex2", "#hex3"]\n'
            f'}}\n'
        )

        try:
            from src.models.claude_client import ClaudeClient

            client = ClaudeClient()
            llm_request = LLMRequest(
                prompt=prompt,
                system_prompt=(
                    "You are an expert at creating image generation prompts for "
                    "professional LinkedIn content. Your prompts produce visually "
                    "striking, on-brand images that complement business posts."
                ),
                max_tokens=1024,
                temperature=0.7,
            )
            response = await client.complete(llm_request)

            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            try:
                image_data = json.loads(content)
                image_data["style"] = style
                image_data["aspect_ratio"] = aspect_ratio
                image_data["generation_api"] = "dall-e-3"
                return image_data
            except json.JSONDecodeError:
                pass

        except Exception as llm_exc:
            await logger.awarning(
                "image_spec_llm_fallback",
                error=str(llm_exc),
            )

        # Fallback: template-based image spec
        return {
            "prompt": (
                f"Professional LinkedIn post illustration about "
                f"{post_content[:100]}. {style_desc}. "
                f"No text, no logos. High quality, {aspect_ratio} aspect ratio."
            ),
            "negative_prompt": "text, watermarks, logos, low quality, blurry, distorted",
            "alt_text": f"Professional illustration for LinkedIn post",
            "color_palette": ["#0077B5", "#FFFFFF", "#F3F2EF"],
            "style": style,
            "aspect_ratio": aspect_ratio,
            "generation_api": "dall-e-3",
        }

    # ── Fallback Generators ───────────────────────────────────────

    @staticmethod
    def _generate_fallback_post(
        topic: str,
        tone: str,
        target_audience: str,
        include_cta: bool,
        post_style: str,
    ) -> dict[str, Any]:
        """Generate a template-based LinkedIn post when LLM is unavailable.

        Args:
            topic: Post topic.
            tone: Writing tone.
            target_audience: Target audience.
            include_cta: Whether to include CTA.
            post_style: Post style.

        Returns:
            Dictionary with template-based post content.
        """
        hook = f"Here's what most people get wrong about {topic}:"
        body = (
            f"\n\nAfter years of working in this space, I've noticed a pattern.\n\n"
            f"The teams that succeed aren't the ones with the most resources.\n"
            f"They're the ones that understand {topic} at a fundamental level.\n\n"
            f"3 things I've learned:\n\n"
            f"1. Start small but start now\n"
            f"2. Measure what matters, not what's easy\n"
            f"3. Iterate based on feedback, not assumptions"
        )
        cta = "\n\nWhat's your experience with this? Drop a comment below." if include_cta else ""
        hashtags = ["AI", "Leadership", "Innovation", "TechCommunity", "FutureOfWork"]

        content = f"{hook}{body}{cta}\n\n" + " ".join(f"#{h}" for h in hashtags[:5])

        return {
            "hook": hook,
            "body": body.strip(),
            "cta": cta.strip(),
            "hashtags": hashtags[:5],
            "content": content,
        }
