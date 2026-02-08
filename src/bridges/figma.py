"""Figma REST API bridge for fetching design metadata from URLs."""

import asyncio
import os
import re
import time
from typing import Any, Optional

import structlog
import httpx

logger = structlog.get_logger(__name__)

# Match Figma URLs:
# https://www.figma.com/file/FILEID/Title?node-id=...
# https://www.figma.com/design/FILEID/Title?node-id=...
# https://www.figma.com/proto/FILEID/...
# https://www.figma.com/board/FILEID/...
FIGMA_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?figma\.com/"
    r"(?:file|design|proto|board)/"
    r"([a-zA-Z0-9]+)"              # file key
    r"(?:/([^?#]*))?"              # optional title slug
    r"(?:\?.*?node-id=([^&#]+))?", # optional node-id
    re.IGNORECASE,
)


def extract_figma_refs(text: str) -> list[dict[str, str]]:
    """Extract Figma file references from text.

    Returns list of dicts with keys: file_key, title, node_id, url
    """
    results = []
    seen = set()

    for match in FIGMA_URL_PATTERN.finditer(text):
        file_key = match.group(1)
        title = (match.group(2) or "").replace("-", " ").strip()
        node_id = match.group(3)

        if file_key not in seen:
            seen.add(file_key)
            results.append({
                "file_key": file_key,
                "title": title,
                "node_id": node_id,
                "url": match.group(0),
            })

    return results


class FigmaBridge:
    """Fetches design metadata from Figma REST API."""

    BASE_URL = "https://api.figma.com/v1"

    # In-memory response cache: key → (timestamp, data)
    _cache: dict[str, tuple[float, Any]] = {}
    _CACHE_TTL = 300  # 5 minutes

    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token or os.getenv("AGENTARMY_FIGMA_TOKEN", "")
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.access_token)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=20.0,
                headers={
                    "X-Figma-Token": self.access_token,
                    "Accept": "application/json",
                },
            )
        return self._client

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        max_retries: int = 3,
        max_wait: float = 60.0,
        **kwargs,
    ) -> httpx.Response:
        """Execute an HTTP request with automatic retry on 429 rate-limit responses.

        Uses exponential backoff, respecting the Retry-After header when present.
        If Retry-After exceeds max_wait, returns immediately (no point waiting days).
        """
        client = self._get_client()
        last_resp: Optional[httpx.Response] = None

        for attempt in range(max_retries + 1):
            resp = await client.request(method, url, **kwargs)
            if resp.status_code != 429:
                return resp

            last_resp = resp
            if attempt >= max_retries:
                break

            # Parse Retry-After header (seconds) or use exponential backoff
            retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            retry_after_val: Optional[float] = None
            if retry_after:
                try:
                    retry_after_val = float(retry_after)
                except (ValueError, TypeError):
                    pass

            if retry_after_val and retry_after_val > max_wait:
                # Figma can return Retry-After of 400,000s (~4.6 days).
                # No point retrying — fail fast with a clear message.
                minutes = int(retry_after_val / 60)
                hours = round(retry_after_val / 3600, 1)
                await logger.awarning(
                    "figma_rate_limit_long_lockout",
                    retry_after=retry_after_val,
                    hours=hours,
                    url=url[:100],
                )
                return resp

            wait = retry_after_val if retry_after_val else (2 ** attempt)
            wait = max(wait, 1.0)
            wait = min(wait, max_wait)

            await logger.awarning(
                "figma_rate_limited",
                attempt=attempt + 1,
                wait_seconds=wait,
                url=url[:100],
            )
            await asyncio.sleep(wait)

        # All retries exhausted — return the last 429 response
        return last_resp  # type: ignore[return-value]

    def _cache_key(self, *parts: str) -> str:
        return "|".join(str(p) for p in parts)

    def _get_cached(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry[0]) < self._CACHE_TTL:
            return entry[1]
        if entry:
            del self._cache[key]
        return None

    def _set_cached(self, key: str, data: Any) -> None:
        self._cache[key] = (time.time(), data)

    async def fetch_file_meta(self, file_key: str, node_id: Optional[str] = None) -> dict[str, Any]:
        """Fetch file metadata and optionally a specific node.

        Args:
            file_key: The Figma file key from the URL.
            node_id: Optional node ID to focus on a specific frame/component.

        Returns:
            Normalized design metadata dict.
        """
        if not self.is_configured:
            return {
                "error": "Figma is not configured. Add your Figma personal access token in Settings.",
                "file_key": file_key,
                "source": "figma",
            }

        try:
            # Check cache first
            ck = self._cache_key("meta", file_key, node_id or "")
            cached = self._get_cached(ck)
            if cached:
                return cached

            # Fetch file info (lightweight — just metadata, no full document tree)
            url = f"{self.BASE_URL}/files/{file_key}"
            params = {"depth": 1}  # Only top-level children for performance

            if node_id:
                # If a node ID is specified, get just that subtree
                decoded_node = node_id.replace("%3A", ":").replace("-", ":")
                params["ids"] = decoded_node

            resp = await self._request_with_retry("GET", url, params=params)

            if resp.status_code == 403:
                return {"error": "Figma access denied — check your token and file permissions.", "file_key": file_key, "source": "figma"}
            if resp.status_code == 404:
                return {"error": f"Figma file {file_key} not found.", "file_key": file_key, "source": "figma"}
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                wait_info = ""
                if retry_after:
                    try:
                        secs = float(retry_after)
                        if secs > 3600:
                            wait_info = f" Lockout: ~{round(secs / 3600, 1)} hours."
                        elif secs > 60:
                            wait_info = f" Lockout: ~{int(secs / 60)} minutes."
                        else:
                            wait_info = f" Retry in {int(secs)}s."
                    except (ValueError, TypeError):
                        pass
                return {"error": f"Figma API rate limit exceeded.{wait_info} Please wait and try again.", "file_key": file_key, "source": "figma"}
            if resp.status_code != 200:
                return {"error": f"Figma API returned {resp.status_code}", "file_key": file_key, "source": "figma"}

            data = resp.json()

            # Extract document structure
            document = data.get("document", {})
            file_name = data.get("name", "Untitled")
            last_modified = data.get("lastModified", "")
            version = data.get("version", "")
            thumbnail_url = data.get("thumbnailUrl", "")

            # Parse pages and top-level frames
            pages = []
            for page in document.get("children", []):
                page_info = {
                    "name": page.get("name", ""),
                    "id": page.get("id", ""),
                    "type": page.get("type", ""),
                    "children_count": len(page.get("children", [])),
                    "frames": [],
                }

                # List top-level frames in each page
                for child in page.get("children", [])[:20]:  # Limit to 20 per page
                    frame_info = {
                        "name": child.get("name", ""),
                        "id": child.get("id", ""),
                        "type": child.get("type", ""),
                    }

                    # Include size if available
                    abs_box = child.get("absoluteBoundingBox")
                    if abs_box:
                        frame_info["width"] = round(abs_box.get("width", 0))
                        frame_info["height"] = round(abs_box.get("height", 0))

                    page_info["frames"].append(frame_info)

                pages.append(page_info)

            # Try to get component info
            components = {}
            for comp_id, comp_data in data.get("components", {}).items():
                components[comp_id] = {
                    "name": comp_data.get("name", ""),
                    "description": comp_data.get("description", ""),
                }

            # Try to get style info
            styles = {}
            for style_id, style_data in data.get("styles", {}).items():
                styles[style_id] = {
                    "name": style_data.get("name", ""),
                    "style_type": style_data.get("styleType", ""),
                    "description": style_data.get("description", ""),
                }

            result = {
                "source": "figma",
                "file_key": file_key,
                "url": f"https://www.figma.com/file/{file_key}",
                "name": file_name,
                "last_modified": last_modified,
                "version": version,
                "thumbnail_url": thumbnail_url,
                "pages": pages,
                "components": components,
                "styles": styles,
                "focused_node_id": node_id,
            }

            # If a specific node was requested, also try to fetch an image
            if node_id:
                image_url = await self._fetch_node_image(file_key, node_id)
                if image_url:
                    result["node_image_url"] = image_url

            self._set_cached(ck, result)
            return result

        except httpx.TimeoutException:
            return {"error": f"Timeout fetching Figma file {file_key}.", "file_key": file_key, "source": "figma"}
        except Exception as exc:
            await logger.aerror("figma_fetch_failed", file_key=file_key, error=str(exc))
            return {"error": f"Failed to fetch from Figma: {str(exc)}", "file_key": file_key, "source": "figma"}

    async def _fetch_node_image(self, file_key: str, node_id: str) -> Optional[str]:
        """Fetch a rendered image URL for a specific node."""
        try:
            decoded_node = node_id.replace("%3A", ":").replace("-", ":")
            url = f"{self.BASE_URL}/images/{file_key}"
            resp = await self._request_with_retry("GET", url, params={
                "ids": decoded_node,
                "format": "png",
                "scale": 2,
            })
            if resp.status_code == 200:
                images = resp.json().get("images", {})
                return images.get(decoded_node)
        except Exception:
            pass
        return None

    def format_for_chat(self, design: dict[str, Any]) -> str:
        """Format Figma design metadata as readable markdown for chat display."""
        if design.get("error"):
            return f"**Figma Error:** {design['error']}"

        lines = []
        lines.append(f"### 🎨 [{design.get('name', 'Figma Design')}]({design.get('url', '')})")
        lines.append("")

        # Meta
        meta = []
        if design.get("last_modified"):
            meta.append(f"**Modified:** {design['last_modified'][:10]}")
        if design.get("pages"):
            meta.append(f"**Pages:** {len(design['pages'])}")
        comp_count = len(design.get("components", {}))
        if comp_count:
            meta.append(f"**Components:** {comp_count}")
        style_count = len(design.get("styles", {}))
        if style_count:
            meta.append(f"**Styles:** {style_count}")
        if meta:
            lines.append(" · ".join(meta))
            lines.append("")

        # Thumbnail
        if design.get("thumbnail_url"):
            lines.append(f"![Thumbnail]({design['thumbnail_url']})")
            lines.append("")

        # Node image (if specific node was requested)
        if design.get("node_image_url"):
            lines.append(f"**Selected Frame:**")
            lines.append(f"![Frame]({design['node_image_url']})")
            lines.append("")

        # Pages and frames
        if design.get("pages"):
            lines.append("**Structure:**")
            for page in design["pages"]:
                lines.append(f"**{page['name']}** ({page.get('children_count', 0)} items)")
                for frame in page.get("frames", [])[:10]:
                    size = ""
                    if frame.get("width") and frame.get("height"):
                        size = f" ({frame['width']}×{frame['height']})"
                    lines.append(f"  • {frame['name']} [{frame.get('type', '')}]{size}")
            lines.append("")

        # Components
        if design.get("components"):
            lines.append(f"**Components ({len(design['components'])}):**")
            for _, comp in list(design["components"].items())[:10]:
                desc = f" — {comp['description']}" if comp.get("description") else ""
                lines.append(f"  • {comp['name']}{desc}")
            if len(design["components"]) > 10:
                lines.append(f"  _...and {len(design['components']) - 10} more_")
            lines.append("")

        return "\n".join(lines)

    def format_for_agent_context(self, design: dict[str, Any]) -> str:
        """Format design metadata as context string for injection into agent prompts."""
        if design.get("error"):
            return ""

        parts = [
            f"FIGMA DESIGN: {design.get('name', 'Untitled')}",
            f"URL: {design.get('url', '')}",
            f"Last Modified: {design.get('last_modified', 'Unknown')}",
        ]

        if design.get("pages"):
            parts.append("\nPages:")
            for page in design["pages"]:
                parts.append(f"  Page: {page['name']} ({page.get('children_count', 0)} items)")
                for frame in page.get("frames", [])[:15]:
                    size = ""
                    if frame.get("width") and frame.get("height"):
                        size = f" ({frame['width']}x{frame['height']})"
                    parts.append(f"    - {frame['name']} [{frame.get('type', '')}]{size}")

        if design.get("components"):
            parts.append(f"\nComponents ({len(design['components'])}):")
            for _, comp in list(design["components"].items())[:15]:
                desc = f" — {comp['description']}" if comp.get("description") else ""
                parts.append(f"  - {comp['name']}{desc}")

        if design.get("styles"):
            parts.append(f"\nStyles ({len(design['styles'])}):")
            for _, style in list(design["styles"].items())[:10]:
                parts.append(f"  - {style['name']} ({style.get('style_type', '')})")

        return "\n".join(parts)

    # ── Design Token Extraction ─────────────────────────────────────

    async def fetch_file_nodes(self, file_key: str, node_ids: Optional[list[str]] = None, depth: int = 2) -> dict[str, Any]:
        """Fetch file document tree with full styling properties.

        Args:
            file_key: Figma file key.
            node_ids: Optional list of node IDs to focus on.
            depth: How deep to traverse (2=lightweight, 3=moderate). Keep low to avoid 429s.

        Returns:
            Raw Figma API response with document tree.
        """
        if not self.is_configured:
            return {"error": "Figma not configured"}

        # Check cache
        ids_str = ",".join(sorted(node_ids or []))
        ck = self._cache_key("nodes", file_key, ids_str, str(depth))
        cached = self._get_cached(ck)
        if cached:
            return cached

        try:
            params: dict[str, Any] = {"depth": depth}
            if node_ids:
                params["ids"] = ",".join(
                    nid.replace("%3A", ":").replace("-", ":") for nid in node_ids
                )

            resp = await self._request_with_retry(
                "GET",
                f"{self.BASE_URL}/files/{file_key}",
                params=params,
                timeout=30.0,
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                wait_info = ""
                if retry_after:
                    try:
                        secs = float(retry_after)
                        if secs > 3600:
                            wait_info = f" Lockout: ~{round(secs / 3600, 1)} hours."
                        elif secs > 60:
                            wait_info = f" Lockout: ~{int(secs / 60)} minutes."
                        else:
                            wait_info = f" Retry in {int(secs)}s."
                    except (ValueError, TypeError):
                        pass
                return {"error": f"Figma API rate limit exceeded.{wait_info} Please wait and try again."}
            if resp.status_code != 200:
                return {"error": f"Figma API {resp.status_code}"}
            data = resp.json()
            self._set_cached(ck, data)
            return data

        except Exception as exc:
            return {"error": str(exc)}

    async def extract_design_tokens(self, file_key: str, node_id: Optional[str] = None) -> dict[str, Any]:
        """Extract design tokens (colors, typography, spacing, effects) from a Figma file.

        Traverses the document tree, collects all unique fills, text styles,
        effects, strokes, border radii, and spacing values used across the design.

        Args:
            file_key: Figma file key.
            node_id: Optional node ID to scope extraction to a specific frame.

        Returns:
            Dict with categorized design tokens.
        """
        node_ids = [node_id] if node_id else None
        # Use depth=3 to balance token coverage vs. API response size.
        # depth=5 pulls multi-MB responses that trigger Figma's 429 rate limits.
        file_data = await self.fetch_file_nodes(file_key, node_ids=node_ids, depth=3)

        if file_data.get("error"):
            return file_data

        tokens: dict[str, Any] = {
            "colors": {},
            "typography": {},
            "effects": [],
            "radii": set(),
            "strokes": [],
            "spacing": set(),
            "grids": [],
        }

        # Published styles metadata from the file
        style_meta = file_data.get("styles", {})

        # Traverse the document tree
        document = file_data.get("document", {})
        self._traverse_node(document, tokens, style_meta)

        # Normalize sets to sorted lists
        tokens["radii"] = sorted(tokens["radii"])
        tokens["spacing"] = sorted(tokens["spacing"])

        # Deduplicate effects
        seen_effects: set[str] = set()
        unique_effects = []
        for eff in tokens["effects"]:
            key = f"{eff.get('type')}_{eff.get('color', '')}_{eff.get('offset', '')}_{eff.get('radius', '')}"
            if key not in seen_effects:
                seen_effects.add(key)
                unique_effects.append(eff)
        tokens["effects"] = unique_effects

        # Deduplicate strokes
        seen_strokes: set[str] = set()
        unique_strokes = []
        for s in tokens["strokes"]:
            key = f"{s.get('color', '')}_{s.get('weight', '')}"
            if key not in seen_strokes:
                seen_strokes.add(key)
                unique_strokes.append(s)
        tokens["strokes"] = unique_strokes

        return {
            "source": "figma",
            "file_key": file_key,
            "file_name": file_data.get("name", "Untitled"),
            "tokens": tokens,
            "style_meta": {sid: sm for sid, sm in style_meta.items()},
        }

    def _traverse_node(self, node: dict[str, Any], tokens: dict[str, Any], style_meta: dict) -> None:
        """Recursively traverse a Figma node tree, extracting styling properties."""
        node_type = node.get("type", "")

        # ── Fills → Colors ──
        fills = node.get("fills", [])
        for fill in fills:
            if fill.get("type") == "SOLID" and fill.get("visible", True):
                color = fill.get("color", {})
                opacity = fill.get("opacity", 1.0)
                hex_color = self._rgba_to_hex(color, opacity)
                rgba = self._rgba_to_string(color, opacity)

                # Check if this fill is a published style
                fill_styles = node.get("styles", {})
                style_id = fill_styles.get("fill") or fill_styles.get("fills")
                style_name = None
                if style_id and style_id in style_meta:
                    style_name = style_meta[style_id].get("name", "")

                key = style_name or hex_color
                if key not in tokens["colors"]:
                    tokens["colors"][key] = {
                        "hex": hex_color,
                        "rgba": rgba,
                        "r": round(color.get("r", 0) * 255),
                        "g": round(color.get("g", 0) * 255),
                        "b": round(color.get("b", 0) * 255),
                        "a": round(opacity, 2),
                        "style_name": style_name,
                        "usage_count": 0,
                    }
                tokens["colors"][key]["usage_count"] += 1

            elif fill.get("type") == "GRADIENT_LINEAR" and fill.get("visible", True):
                stops = fill.get("gradientStops", [])
                if stops:
                    for stop in stops:
                        color = stop.get("color", {})
                        hex_c = self._rgba_to_hex(color, color.get("a", 1.0))
                        key = hex_c
                        if key not in tokens["colors"]:
                            tokens["colors"][key] = {
                                "hex": hex_c,
                                "rgba": self._rgba_to_string(color, color.get("a", 1.0)),
                                "r": round(color.get("r", 0) * 255),
                                "g": round(color.get("g", 0) * 255),
                                "b": round(color.get("b", 0) * 255),
                                "a": round(color.get("a", 1.0), 2),
                                "style_name": None,
                                "usage_count": 0,
                                "from_gradient": True,
                            }
                        tokens["colors"][key]["usage_count"] += 1

        # ── Text → Typography ──
        if node_type == "TEXT":
            style = node.get("style", {})
            if style:
                font_family = style.get("fontFamily", "")
                font_size = style.get("fontSize", 0)
                font_weight = style.get("fontWeight", 400)
                line_height_px = style.get("lineHeightPx")
                line_height_pct = style.get("lineHeightPercentFontSize")
                letter_spacing = style.get("letterSpacing", 0)
                text_align = style.get("textAlignHorizontal", "LEFT")
                text_decoration = style.get("textDecoration", "NONE")
                text_case = style.get("textCase", "ORIGINAL")

                # Check for text style reference
                text_styles = node.get("styles", {})
                style_id = text_styles.get("text")
                style_name = None
                if style_id and style_id in style_meta:
                    style_name = style_meta[style_id].get("name", "")

                key = style_name or f"{font_family}/{font_size}/{font_weight}"
                if key not in tokens["typography"]:
                    typo: dict[str, Any] = {
                        "font_family": font_family,
                        "font_size": font_size,
                        "font_weight": font_weight,
                        "letter_spacing": round(letter_spacing, 2) if letter_spacing else 0,
                        "text_align": text_align,
                        "text_decoration": text_decoration,
                        "text_case": text_case,
                        "style_name": style_name,
                        "usage_count": 0,
                    }
                    if line_height_px:
                        typo["line_height_px"] = round(line_height_px, 1)
                    if line_height_pct:
                        typo["line_height_percent"] = round(line_height_pct, 1)
                    tokens["typography"][key] = typo
                tokens["typography"][key]["usage_count"] += 1

        # ── Effects (shadows, blurs) ──
        effects = node.get("effects", [])
        for eff in effects:
            if not eff.get("visible", True):
                continue
            eff_type = eff.get("type", "")
            parsed: dict[str, Any] = {"type": eff_type}

            if eff_type in ("DROP_SHADOW", "INNER_SHADOW"):
                color = eff.get("color", {})
                parsed["color"] = self._rgba_to_string(color, color.get("a", 1.0))
                parsed["hex"] = self._rgba_to_hex(color, color.get("a", 1.0))
                offset = eff.get("offset", {})
                parsed["offset_x"] = round(offset.get("x", 0), 1)
                parsed["offset_y"] = round(offset.get("y", 0), 1)
                parsed["radius"] = round(eff.get("radius", 0), 1)
                parsed["spread"] = round(eff.get("spread", 0), 1)
                # Generate CSS
                inset = "inset " if eff_type == "INNER_SHADOW" else ""
                parsed["css"] = f"{inset}{parsed['offset_x']}px {parsed['offset_y']}px {parsed['radius']}px {parsed['spread']}px {parsed['color']}"

            elif eff_type in ("LAYER_BLUR", "BACKGROUND_BLUR"):
                parsed["radius"] = round(eff.get("radius", 0), 1)
                parsed["css"] = f"blur({parsed['radius']}px)"

            tokens["effects"].append(parsed)

        # ── Border radius ──
        corner_radius = node.get("cornerRadius")
        if corner_radius and corner_radius > 0:
            tokens["radii"].add(round(corner_radius, 1))
        # Individual corners
        for corner_key in ("topLeftRadius", "topRightRadius", "bottomLeftRadius", "bottomRightRadius"):
            cr = node.get(corner_key)
            if cr and cr > 0:
                tokens["radii"].add(round(cr, 1))

        # ── Strokes ──
        strokes = node.get("strokes", [])
        stroke_weight = node.get("strokeWeight")
        for stroke in strokes:
            if stroke.get("type") == "SOLID" and stroke.get("visible", True):
                color = stroke.get("color", {})
                tokens["strokes"].append({
                    "color": self._rgba_to_hex(color, stroke.get("opacity", 1.0)),
                    "rgba": self._rgba_to_string(color, stroke.get("opacity", 1.0)),
                    "weight": stroke_weight or 1,
                })

        # ── Auto-layout spacing ──
        if node.get("layoutMode"):
            item_spacing = node.get("itemSpacing")
            if item_spacing and item_spacing > 0:
                tokens["spacing"].add(round(item_spacing))
            for pad_key in ("paddingLeft", "paddingRight", "paddingTop", "paddingBottom"):
                pad = node.get(pad_key)
                if pad and pad > 0:
                    tokens["spacing"].add(round(pad))

        # ── Layout grids ──
        for grid in node.get("layoutGrids", []):
            if grid.get("visible", True):
                tokens["grids"].append({
                    "pattern": grid.get("pattern", ""),
                    "section_size": grid.get("sectionSize"),
                    "count": grid.get("count"),
                    "gutter_size": grid.get("gutterSize"),
                    "offset": grid.get("offset"),
                    "alignment": grid.get("alignment", ""),
                })

        # Recurse into children
        for child in node.get("children", []):
            self._traverse_node(child, tokens, style_meta)

    # ── Output Generators ────────────────────────────────────────────

    def tokens_to_css(self, token_data: dict[str, Any]) -> str:
        """Convert extracted tokens into CSS custom properties.

        Args:
            token_data: Output from extract_design_tokens().

        Returns:
            CSS string with :root variables.
        """
        tokens = token_data.get("tokens", {})
        lines = [f"/* Design Tokens — extracted from Figma: {token_data.get('file_name', '')} */", ""]
        lines.append(":root {")

        # Colors
        if tokens.get("colors"):
            lines.append("  /* Colors */")
            for name, c in sorted(tokens["colors"].items(), key=lambda x: x[1].get("usage_count", 0), reverse=True):
                css_name = self._to_css_var_name(name)
                lines.append(f"  --color-{css_name}: {c['hex']};")
            lines.append("")

        # Typography as separate variables
        if tokens.get("typography"):
            lines.append("  /* Typography */")
            font_families = set()
            font_sizes = set()
            for name, t in tokens["typography"].items():
                if t.get("font_family"):
                    font_families.add(t["font_family"])
                if t.get("font_size"):
                    font_sizes.add(t["font_size"])

            for i, ff in enumerate(sorted(font_families)):
                lines.append(f"  --font-family-{i}: '{ff}';")
            for fs in sorted(font_sizes):
                lines.append(f"  --font-size-{int(fs)}: {fs}px;")
            lines.append("")

        # Border radii
        if tokens.get("radii"):
            lines.append("  /* Border Radius */")
            for r in tokens["radii"]:
                lines.append(f"  --radius-{int(r)}: {r}px;")
            lines.append("")

        # Spacing
        if tokens.get("spacing"):
            lines.append("  /* Spacing */")
            for s in tokens["spacing"]:
                lines.append(f"  --space-{int(s)}: {s}px;")
            lines.append("")

        lines.append("}")
        lines.append("")

        # Shadow utility classes
        if tokens.get("effects"):
            shadows = [e for e in tokens["effects"] if e["type"] in ("DROP_SHADOW", "INNER_SHADOW")]
            if shadows:
                lines.append("/* Shadows */")
                for i, s in enumerate(shadows):
                    lines.append(f".shadow-{i} {{ box-shadow: {s['css']}; }}")
                lines.append("")

        # Typography utility classes
        if tokens.get("typography"):
            lines.append("/* Typography */")
            for name, t in tokens["typography"].items():
                cls = self._to_css_var_name(name)
                props = []
                if t.get("font_family"):
                    props.append(f"  font-family: '{t['font_family']}';")
                if t.get("font_size"):
                    props.append(f"  font-size: {t['font_size']}px;")
                if t.get("font_weight"):
                    props.append(f"  font-weight: {t['font_weight']};")
                if t.get("line_height_px"):
                    props.append(f"  line-height: {t['line_height_px']}px;")
                if t.get("letter_spacing"):
                    props.append(f"  letter-spacing: {t['letter_spacing']}px;")
                if props:
                    lines.append(f".text-{cls} {{")
                    lines.extend(props)
                    lines.append("}")
            lines.append("")

        return "\n".join(lines)

    def tokens_to_tailwind(self, token_data: dict[str, Any]) -> str:
        """Convert extracted tokens into a Tailwind CSS config extension.

        Args:
            token_data: Output from extract_design_tokens().

        Returns:
            JavaScript string for tailwind.config.js theme.extend.
        """
        tokens = token_data.get("tokens", {})
        config: dict[str, Any] = {"colors": {}, "fontSize": {}, "fontFamily": {}, "borderRadius": {}, "spacing": {}, "boxShadow": {}}

        # Colors
        for name, c in tokens.get("colors", {}).items():
            key = self._to_tailwind_key(name)
            config["colors"][key] = c["hex"]

        # Typography
        families: set[str] = set()
        for t in tokens.get("typography", {}).values():
            if t.get("font_family"):
                families.add(t["font_family"])
            if t.get("font_size"):
                config["fontSize"][f"{int(t['font_size'])}"] = f"{t['font_size']}px"

        for ff in sorted(families):
            key = ff.lower().replace(" ", "-")
            config["fontFamily"][key] = [f"'{ff}'", "sans-serif"]

        # Radii
        for r in tokens.get("radii", []):
            config["borderRadius"][f"{int(r)}"] = f"{r}px"

        # Spacing
        for s in tokens.get("spacing", []):
            config["spacing"][f"{int(s)}"] = f"{s}px"

        # Shadows
        for i, eff in enumerate(tokens.get("effects", [])):
            if eff.get("css") and eff["type"] in ("DROP_SHADOW", "INNER_SHADOW"):
                config["boxShadow"][f"figma-{i}"] = eff["css"]

        # Build JS output
        import json
        lines = [
            f"// Tailwind config — extracted from Figma: {token_data.get('file_name', '')}",
            "// Add to tailwind.config.js → theme.extend",
            "",
            "module.exports = {",
            "  theme: {",
            "    extend: {",
        ]

        for section, values in config.items():
            if values:
                lines.append(f"      {section}: {{")
                for k, v in values.items():
                    if isinstance(v, list):
                        lines.append(f"        '{k}': {json.dumps(v)},")
                    else:
                        lines.append(f"        '{k}': '{v}',")
                lines.append("      },")

        lines.extend(["    },", "  },", "};", ""])
        return "\n".join(lines)

    def tokens_to_json(self, token_data: dict[str, Any]) -> dict[str, Any]:
        """Return design tokens as a clean JSON-serializable dict."""
        import copy
        tokens = copy.deepcopy(token_data.get("tokens", {}))
        # sets → lists for JSON serialization
        if "radii" in tokens and isinstance(tokens["radii"], set):
            tokens["radii"] = sorted(tokens["radii"])
        if "spacing" in tokens and isinstance(tokens["spacing"], set):
            tokens["spacing"] = sorted(tokens["spacing"])
        return {
            "file_name": token_data.get("file_name", ""),
            "file_key": token_data.get("file_key", ""),
            **tokens,
        }

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _rgba_to_hex(color: dict, opacity: float = 1.0) -> str:
        r = round(color.get("r", 0) * 255)
        g = round(color.get("g", 0) * 255)
        b = round(color.get("b", 0) * 255)
        if opacity < 1.0:
            a = round(opacity * 255)
            return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _rgba_to_string(color: dict, opacity: float = 1.0) -> str:
        r = round(color.get("r", 0) * 255)
        g = round(color.get("g", 0) * 255)
        b = round(color.get("b", 0) * 255)
        if opacity < 1.0:
            return f"rgba({r}, {g}, {b}, {round(opacity, 2)})"
        return f"rgb({r}, {g}, {b})"

    @staticmethod
    def _to_css_var_name(name: str) -> str:
        """Convert a style name like 'Brand/Primary' into a CSS-friendly slug."""
        slug = name.lower().strip()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = slug.strip("-")
        return slug or "unnamed"

    @staticmethod
    def _to_tailwind_key(name: str) -> str:
        """Convert a style name into a Tailwind-friendly key."""
        slug = name.lower().strip()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = slug.strip("-")
        return slug or "unnamed"

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
