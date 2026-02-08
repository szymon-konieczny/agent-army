"""Social media designer agent — generates images for LinkedIn, Twitter, etc."""

import json
from datetime import datetime, timezone
from typing import Any

import structlog

from src.core.agent_base import AgentCapability, AgentIdentity, AgentState, BaseAgent

logger = structlog.get_logger(__name__)

# ── Social media image presets ──────────────────────────────────
SOCIAL_PRESETS = {
    "linkedin_post": {
        "label": "LinkedIn Post Image",
        "aspect_ratio": "4:3",
        "width": 1200,
        "height": 627,
        "guidance": (
            "Professional, clean, modern design. "
            "Strong visual hierarchy. Corporate-friendly colors. "
            "Minimal text overlay if any. High contrast."
        ),
    },
    "linkedin_banner": {
        "label": "LinkedIn Profile Banner",
        "aspect_ratio": "16:9",
        "width": 1584,
        "height": 396,
        "guidance": (
            "Wide panoramic composition. Professional, sleek. "
            "Brand-aligned colors. Subtle gradient or abstract patterns. "
            "Space for profile photo overlay on the left side."
        ),
    },
    "linkedin_article": {
        "label": "LinkedIn Article Cover",
        "aspect_ratio": "16:9",
        "width": 1280,
        "height": 720,
        "guidance": (
            "Editorial style hero image. Cinematic, high quality. "
            "Should convey the article topic visually. "
            "Rich colors, dramatic lighting."
        ),
    },
    "instagram_post": {
        "label": "Instagram Post",
        "aspect_ratio": "1:1",
        "width": 1080,
        "height": 1080,
        "guidance": (
            "Eye-catching, vibrant, scroll-stopping. "
            "Clean composition. Bold colors. "
            "Optimized for mobile viewing."
        ),
    },
    "instagram_story": {
        "label": "Instagram/TikTok Story",
        "aspect_ratio": "9:16",
        "width": 1080,
        "height": 1920,
        "guidance": (
            "Full-screen vertical format. Immersive, dynamic. "
            "Strong focal point in the center. "
            "Room for text/stickers at top and bottom."
        ),
    },
    "twitter_post": {
        "label": "Twitter/X Post Image",
        "aspect_ratio": "16:9",
        "width": 1200,
        "height": 675,
        "guidance": (
            "Attention-grabbing, clear at small sizes. "
            "High contrast. Works well as a thumbnail. "
            "Simple composition, bold visual."
        ),
    },
    "twitter_header": {
        "label": "Twitter/X Header",
        "aspect_ratio": "3:1",
        "width": 1500,
        "height": 500,
        "guidance": (
            "Wide banner format. Clean and professional. "
            "Central focus area. Subtle branding. "
            "Works with profile photo overlap on the left."
        ),
    },
    "facebook_post": {
        "label": "Facebook Post Image",
        "aspect_ratio": "16:9",
        "width": 1200,
        "height": 630,
        "guidance": (
            "Engaging, shareable visual. Clear message. "
            "Works at small thumbnail size in feed. "
            "Professional but approachable."
        ),
    },
    "youtube_thumbnail": {
        "label": "YouTube Thumbnail",
        "aspect_ratio": "16:9",
        "width": 1280,
        "height": 720,
        "guidance": (
            "High-impact, click-worthy. Bold colors, expressive. "
            "Large readable text if any. Strong contrast. "
            "Faces and emotions work well."
        ),
    },
    "og_image": {
        "label": "Open Graph / Social Share",
        "aspect_ratio": "16:9",
        "width": 1200,
        "height": 630,
        "guidance": (
            "Clean, representative preview image. "
            "Works as link preview across all platforms. "
            "Professional with clear visual message."
        ),
    },
}


class DesignerAgent(BaseAgent):
    """Social media image designer agent using Gemini Nano Banana.

    Responsibilities:
    - Generates professional social media images (LinkedIn, Twitter, Instagram, etc.)
    - Creates platform-specific images with correct aspect ratios
    - Designs LinkedIn post images, profile banners, and article covers
    - Produces YouTube thumbnails, Instagram stories, Twitter headers
    - Crafts Open Graph images for link previews
    - Provides art direction and image prompts tailored to each platform

    Capabilities:
    - generate_social_image: Generate an image for a specific social platform
    - create_image_set: Generate images across multiple platforms at once
    - suggest_visuals: Recommend image concepts based on content/text
    """

    def __init__(
        self,
        agent_id: str = "designer-social",
        name: str = "Designer Agent",
        role: str = "designer",
    ) -> None:
        identity = AgentIdentity(
            id=agent_id,
            name=name,
            role=role,
            security_level=2,
            capabilities=[
                AgentCapability(
                    name="generate_social_image",
                    version="1.0.0",
                    description="Generate a social media image for a specific platform",
                    parameters={
                        "prompt": "str",
                        "platform": "str",
                        "style": "str",
                    },
                ),
                AgentCapability(
                    name="create_image_set",
                    version="1.0.0",
                    description="Generate images for multiple platforms from one concept",
                    parameters={
                        "prompt": "str",
                        "platforms": "list[str]",
                    },
                ),
                AgentCapability(
                    name="suggest_visuals",
                    version="1.0.0",
                    description="Recommend image concepts based on post content",
                    parameters={
                        "content": "str",
                        "platform": "str",
                    },
                ),
            ],
        )
        super().__init__(identity)
        self._images_generated: list[dict[str, Any]] = []

    async def startup(self) -> None:
        await super().startup()
        await logger.ainfo("designer_startup", agent_id=self.identity.id)

    async def shutdown(self) -> None:
        await logger.ainfo(
            "designer_shutdown",
            agent_id=self.identity.id,
            images_generated=len(self._images_generated),
        )
        await super().shutdown()

    def _get_system_context(self) -> str:
        return (
            "You are a professional social media visual designer and art director. "
            "You specialize in creating stunning, platform-optimized images for "
            "LinkedIn, Twitter/X, Instagram, YouTube, and Facebook. "
            "You understand each platform's image requirements, best practices, "
            "and what makes content visually engaging.\n\n"
            "KEY CAPABILITIES:\n"
            "- You can GENERATE IMAGES using the built-in Gemini Nano Banana API endpoint.\n"
            "- To generate an image, use a bash command:\n"
            '  curl -s -X POST $AGENTARMY_API_URL/api/generate-image '
            '-H "Content-Type: application/json" '
            "-d '{\"prompt\": \"YOUR DETAILED PROMPT\", \"aspect_ratio\": \"RATIO\"}'\n"
            "- Then display the result with markdown: ![description](URL_FROM_RESPONSE)\n\n"
            "PLATFORM PRESETS:\n"
            + "\n".join(
                f"- **{p['label']}**: {info['aspect_ratio']} ({info['width']}x{info['height']}px) — {info['guidance'][:60]}..."
                for key, info in SOCIAL_PRESETS.items()
                if (p := info)
            )
            + "\n\n"
            "DESIGN PRINCIPLES:\n"
            "- Always use the correct aspect ratio for the target platform\n"
            "- Write detailed, specific image prompts (style, mood, colors, composition)\n"
            "- Suggest multiple concepts when asked, then generate the chosen one\n"
            "- Consider brand consistency across platforms\n"
            "- Prioritize professional, high-quality aesthetics\n"
            "- Always show the generated image in your response using markdown\n"
        )

    async def process_task(self, task: dict[str, Any]) -> dict[str, Any]:
        task_type = task.get("type", "unknown")
        task_id = task.get("id", "unknown")

        await logger.ainfo(
            "designer_processing_task",
            task_id=task_id,
            task_type=task_type,
        )

        reasoning = await self.think(task)
        task.setdefault("context", {})["_reasoning"] = reasoning.conclusion

        try:
            if task_type == "generate_social_image":
                result = await self._handle_generate_image(task)
            elif task_type == "create_image_set":
                result = await self._handle_image_set(task)
            elif task_type == "suggest_visuals":
                result = await self._handle_suggest_visuals(task)
            else:
                return await self._handle_chat_message(task)
            result["reasoning"] = reasoning.to_audit_dict()
            return result
        except Exception as exc:
            await logger.aerror(
                "designer_task_error", task_id=task_id, error=str(exc),
            )
            raise

    async def _handle_generate_image(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate a social media image via the API."""
        params = task.get("parameters", {})
        prompt = params.get("prompt", task.get("description", ""))
        platform = params.get("platform", "linkedin_post")
        style = params.get("style", "")

        preset = SOCIAL_PRESETS.get(platform, SOCIAL_PRESETS["linkedin_post"])

        # Build enhanced prompt with platform guidance
        full_prompt = (
            f"{prompt}. "
            f"Style: {preset['guidance']} "
            f"{style}. "
            f"Dimensions: {preset['width']}x{preset['height']}px. "
            f"Professional quality, 4K, sharp details."
        )

        self._images_generated.append({
            "prompt": prompt,
            "platform": platform,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return {
            "status": "completed",
            "type": "generate_social_image",
            "prompt": full_prompt,
            "aspect_ratio": preset["aspect_ratio"],
            "platform": platform,
            "preset": preset,
            "message": (
                f"Generated image prompt for {preset['label']}. "
                f"Use the /api/generate-image endpoint with aspect_ratio='{preset['aspect_ratio']}'."
            ),
        }

    async def _handle_image_set(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate prompts for multiple platforms."""
        params = task.get("parameters", {})
        prompt = params.get("prompt", task.get("description", ""))
        platforms = params.get("platforms", ["linkedin_post", "twitter_post", "instagram_post"])

        results = []
        for platform in platforms:
            preset = SOCIAL_PRESETS.get(platform, SOCIAL_PRESETS["linkedin_post"])
            results.append({
                "platform": platform,
                "label": preset["label"],
                "aspect_ratio": preset["aspect_ratio"],
                "dimensions": f"{preset['width']}x{preset['height']}",
                "enhanced_prompt": f"{prompt}. {preset['guidance']}",
            })

        return {
            "status": "completed",
            "type": "create_image_set",
            "platforms": results,
        }

    async def _handle_suggest_visuals(self, task: dict[str, Any]) -> dict[str, Any]:
        """Suggest image concepts based on content."""
        params = task.get("parameters", {})
        content = params.get("content", task.get("description", ""))
        platform = params.get("platform", "linkedin_post")

        # Use LLM to generate visual suggestions
        return await self._handle_chat_message(task)
