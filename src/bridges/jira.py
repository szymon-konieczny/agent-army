"""Jira Cloud REST API bridge for fetching issue details from URLs."""

import os
import re
from typing import Any, Optional
from datetime import datetime

import structlog
import httpx

logger = structlog.get_logger(__name__)

# Match Jira URLs: https://xyz.atlassian.net/browse/PROJ-123 or /jira/software/projects/PROJ/boards/.../PROJ-123
JIRA_URL_PATTERN = re.compile(
    r"https?://([a-zA-Z0-9_-]+)\.atlassian\.net"
    r"(?:/browse/|.*?/boards?/\d+.*?/)"
    r"([A-Z][A-Z0-9_]+-\d+)",
    re.IGNORECASE,
)

# Also match simpler issue key in path
JIRA_URL_ALT_PATTERN = re.compile(
    r"https?://([a-zA-Z0-9_-]+)\.atlassian\.net/.*?([A-Z][A-Z0-9_]+-\d+)",
    re.IGNORECASE,
)


def extract_jira_refs(text: str) -> list[dict[str, str]]:
    """Extract Jira issue references from text.

    Returns list of dicts with keys: domain, issue_key, url
    """
    results = []
    seen = set()

    for pattern in [JIRA_URL_PATTERN, JIRA_URL_ALT_PATTERN]:
        for match in pattern.finditer(text):
            domain = match.group(1)
            issue_key = match.group(2).upper()
            if issue_key not in seen:
                seen.add(issue_key)
                results.append({
                    "domain": domain,
                    "issue_key": issue_key,
                    "url": f"https://{domain}.atlassian.net/browse/{issue_key}",
                })

    return results


class JiraBridge:
    """Fetches issue details from Jira Cloud REST API."""

    def __init__(
        self,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        self.email = email or os.getenv("AGENTARMY_JIRA_EMAIL", "")
        self.api_token = api_token or os.getenv("AGENTARMY_JIRA_API_TOKEN", "")
        self.domain = domain or os.getenv("AGENTARMY_JIRA_DOMAIN", "")
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.email and self.api_token)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                auth=(self.email, self.api_token),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def fetch_issue(self, domain: str, issue_key: str) -> dict[str, Any]:
        """Fetch issue details from Jira Cloud.

        Args:
            domain: The Atlassian domain prefix (e.g. 'mycompany' for mycompany.atlassian.net)
            issue_key: The issue key (e.g. 'PROJ-123')

        Returns:
            Normalized issue details dict.
        """
        if not self.is_configured:
            return {
                "error": "Jira is not configured. Add your Jira email and API token in Settings.",
                "issue_key": issue_key,
                "source": "jira",
            }

        base_url = f"https://{domain}.atlassian.net"
        url = f"{base_url}/rest/api/3/issue/{issue_key}"

        try:
            client = self._get_client()
            resp = await client.get(
                url,
                params={
                    "fields": "summary,status,assignee,reporter,priority,issuetype,"
                              "description,comment,subtasks,parent,labels,created,updated,"
                              "fixVersions,components,resolution,sprint",
                },
            )

            if resp.status_code == 401:
                return {"error": "Jira authentication failed — check your email and API token.", "issue_key": issue_key, "source": "jira"}
            if resp.status_code == 404:
                return {"error": f"Issue {issue_key} not found.", "issue_key": issue_key, "source": "jira"}
            if resp.status_code != 200:
                return {"error": f"Jira API returned {resp.status_code}", "issue_key": issue_key, "source": "jira"}

            data = resp.json()
            fields = data.get("fields", {})

            # Extract description text from Atlassian Document Format (ADF)
            description = self._adf_to_text(fields.get("description"))

            # Extract comments (last 5)
            comments_data = fields.get("comment", {})
            comments_list = comments_data.get("comments", []) if isinstance(comments_data, dict) else []
            recent_comments = []
            for c in comments_list[-5:]:
                recent_comments.append({
                    "author": (c.get("author", {}) or {}).get("displayName", "Unknown"),
                    "body": self._adf_to_text(c.get("body")),
                    "created": c.get("created", ""),
                })

            # Extract subtasks
            subtasks = []
            for st in fields.get("subtasks", []):
                st_fields = st.get("fields", {})
                subtasks.append({
                    "key": st.get("key", ""),
                    "summary": st_fields.get("summary", ""),
                    "status": (st_fields.get("status", {}) or {}).get("name", ""),
                })

            return {
                "source": "jira",
                "issue_key": issue_key,
                "url": f"{base_url}/browse/{issue_key}",
                "summary": fields.get("summary", ""),
                "status": (fields.get("status", {}) or {}).get("name", ""),
                "issue_type": (fields.get("issuetype", {}) or {}).get("name", ""),
                "priority": (fields.get("priority", {}) or {}).get("name", ""),
                "assignee": (fields.get("assignee", {}) or {}).get("displayName", "Unassigned"),
                "reporter": (fields.get("reporter", {}) or {}).get("displayName", "Unknown"),
                "labels": fields.get("labels", []),
                "components": [c.get("name", "") for c in (fields.get("components") or [])],
                "description": description,
                "comments": recent_comments,
                "subtasks": subtasks,
                "parent_key": (fields.get("parent", {}) or {}).get("key"),
                "created": fields.get("created", ""),
                "updated": fields.get("updated", ""),
                "resolution": (fields.get("resolution", {}) or {}).get("name"),
            }

        except httpx.TimeoutException:
            return {"error": f"Timeout fetching {issue_key} from Jira.", "issue_key": issue_key, "source": "jira"}
        except Exception as exc:
            await logger.aerror("jira_fetch_failed", issue_key=issue_key, error=str(exc))
            return {"error": f"Failed to fetch from Jira: {str(exc)}", "issue_key": issue_key, "source": "jira"}

    def _adf_to_text(self, adf_node: Any, depth: int = 0) -> str:
        """Convert Atlassian Document Format (ADF) JSON to plain text."""
        if adf_node is None:
            return ""
        if isinstance(adf_node, str):
            return adf_node

        if not isinstance(adf_node, dict):
            return ""

        node_type = adf_node.get("type", "")
        text_parts = []

        # Handle text nodes
        if node_type == "text":
            return adf_node.get("text", "")

        # Handle content array
        for child in adf_node.get("content", []):
            text_parts.append(self._adf_to_text(child, depth + 1))

        joined = "".join(text_parts)

        # Add formatting based on node type
        if node_type == "paragraph":
            return joined + "\n"
        elif node_type == "heading":
            level = adf_node.get("attrs", {}).get("level", 1)
            return "#" * level + " " + joined + "\n"
        elif node_type == "bulletList":
            return joined
        elif node_type == "orderedList":
            return joined
        elif node_type == "listItem":
            return "• " + joined
        elif node_type == "codeBlock":
            lang = adf_node.get("attrs", {}).get("language", "")
            return f"```{lang}\n{joined}```\n"
        elif node_type == "blockquote":
            return "> " + joined
        elif node_type == "rule":
            return "---\n"

        return joined

    def format_for_chat(self, issue: dict[str, Any]) -> str:
        """Format a Jira issue dict as a readable markdown string for chat display."""
        if issue.get("error"):
            return f"**Jira Error:** {issue['error']}"

        lines = []
        lines.append(f"### [{issue['issue_key']}]({issue.get('url', '')}) — {issue.get('summary', '')}")
        lines.append("")

        # Status row
        meta = []
        if issue.get("status"):
            meta.append(f"**Status:** {issue['status']}")
        if issue.get("issue_type"):
            meta.append(f"**Type:** {issue['issue_type']}")
        if issue.get("priority"):
            meta.append(f"**Priority:** {issue['priority']}")
        if meta:
            lines.append(" · ".join(meta))

        # People
        people = []
        if issue.get("assignee") and issue["assignee"] != "Unassigned":
            people.append(f"**Assignee:** {issue['assignee']}")
        if issue.get("reporter"):
            people.append(f"**Reporter:** {issue['reporter']}")
        if people:
            lines.append(" · ".join(people))

        # Labels / components
        tags = []
        if issue.get("labels"):
            tags.extend(issue["labels"])
        if issue.get("components"):
            tags.extend(issue["components"])
        if tags:
            lines.append("**Tags:** " + ", ".join(tags))

        lines.append("")

        # Description
        if issue.get("description"):
            desc = issue["description"]
            if len(desc) > 800:
                desc = desc[:800] + "..."
            lines.append("**Description:**")
            lines.append(desc)
            lines.append("")

        # Subtasks
        if issue.get("subtasks"):
            lines.append("**Subtasks:**")
            for st in issue["subtasks"]:
                status_icon = "✓" if st.get("status", "").lower() in ("done", "closed", "resolved") else "○"
                lines.append(f"  {status_icon} **{st['key']}** — {st['summary']} ({st.get('status', '')})")
            lines.append("")

        # Recent comments
        if issue.get("comments"):
            lines.append(f"**Recent Comments ({len(issue['comments'])}):**")
            for c in issue["comments"][-3:]:
                body = c.get("body", "")
                if len(body) > 200:
                    body = body[:200] + "..."
                lines.append(f"> **{c.get('author', 'Unknown')}:** {body}")
            lines.append("")

        return "\n".join(lines)

    def format_for_agent_context(self, issue: dict[str, Any]) -> str:
        """Format issue as context string for injection into agent prompts."""
        if issue.get("error"):
            return ""

        parts = [
            f"JIRA ISSUE: {issue['issue_key']} — {issue.get('summary', '')}",
            f"Status: {issue.get('status', 'Unknown')}  |  Type: {issue.get('issue_type', '')}  |  Priority: {issue.get('priority', '')}",
            f"Assignee: {issue.get('assignee', 'Unassigned')}  |  Reporter: {issue.get('reporter', '')}",
        ]

        if issue.get("labels") or issue.get("components"):
            tags = (issue.get("labels", []) or []) + (issue.get("components", []) or [])
            parts.append(f"Labels/Components: {', '.join(tags)}")

        if issue.get("description"):
            desc = issue["description"]
            if len(desc) > 1500:
                desc = desc[:1500] + "..."
            parts.append(f"\nDescription:\n{desc}")

        if issue.get("subtasks"):
            parts.append("\nSubtasks:")
            for st in issue["subtasks"]:
                parts.append(f"  - {st['key']}: {st['summary']} [{st.get('status', '')}]")

        if issue.get("comments"):
            parts.append(f"\nRecent Comments:")
            for c in issue["comments"][-3:]:
                body = c.get("body", "")
                if len(body) > 300:
                    body = body[:300] + "..."
                parts.append(f"  {c.get('author', 'Unknown')}: {body}")

        return "\n".join(parts)

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
