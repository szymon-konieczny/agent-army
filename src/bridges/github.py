"""GitHub API bridge for cloning template repos and fetching boilerplate code."""

import os
import re
import shutil
import subprocess
from typing import Any, Optional

import structlog
import httpx

logger = structlog.get_logger(__name__)

# Match GitHub URLs:
# https://github.com/owner/repo
# https://github.com/owner/repo/tree/branch/path
GITHUB_URL_PATTERN = re.compile(
    r"https?://github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)(?:/tree/([^/]+)(?:/(.+))?)?",
    re.IGNORECASE,
)


def extract_github_refs(text: str) -> list[dict[str, str]]:
    """Extract GitHub repo references from text.

    Returns list of dicts with keys: owner, repo, branch, path, url
    """
    results = []
    seen = set()

    for match in GITHUB_URL_PATTERN.finditer(text):
        owner = match.group(1)
        repo = match.group(2).rstrip(".git")
        branch = match.group(3)
        path = match.group(4)
        key = f"{owner}/{repo}"

        if key not in seen:
            seen.add(key)
            results.append({
                "owner": owner,
                "repo": repo,
                "branch": branch,
                "path": path,
                "url": f"https://github.com/{owner}/{repo}",
            })

    return results


class GitHubBridge:
    """Fetches repo metadata and clones template repos from GitHub."""

    BASE_URL = "https://api.github.com"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("AGENTARMY_GITHUB_TOKEN", "")
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.token)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"Accept": "application/vnd.github+json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            self._client = httpx.AsyncClient(
                timeout=15.0,
                headers=headers,
            )
        return self._client

    async def fetch_repo_info(self, owner: str, repo: str) -> dict[str, Any]:
        """Fetch repository metadata from GitHub.

        Args:
            owner: GitHub username or org
            repo: Repository name

        Returns:
            Normalized repo metadata dict.
        """
        try:
            client = self._get_client()
            resp = await client.get(f"{self.BASE_URL}/repos/{owner}/{repo}")

            if resp.status_code == 404:
                return {"error": f"Repository {owner}/{repo} not found.", "source": "github"}
            if resp.status_code == 403:
                return {"error": "GitHub API rate limit or authentication issue.", "source": "github"}
            if resp.status_code != 200:
                return {"error": f"GitHub API returned {resp.status_code}", "source": "github"}

            data = resp.json()

            # Fetch languages
            languages = {}
            try:
                lang_resp = await client.get(f"{self.BASE_URL}/repos/{owner}/{repo}/languages")
                if lang_resp.status_code == 200:
                    languages = lang_resp.json()
            except Exception:
                pass

            # Fetch top-level directory contents
            tree_contents = []
            try:
                tree_resp = await client.get(f"{self.BASE_URL}/repos/{owner}/{repo}/contents")
                if tree_resp.status_code == 200:
                    for item in tree_resp.json()[:30]:
                        tree_contents.append({
                            "name": item.get("name", ""),
                            "type": item.get("type", ""),
                            "size": item.get("size", 0),
                        })
            except Exception:
                pass

            return {
                "source": "github",
                "owner": owner,
                "repo": repo,
                "url": data.get("html_url", f"https://github.com/{owner}/{repo}"),
                "name": data.get("full_name", f"{owner}/{repo}"),
                "description": data.get("description", ""),
                "stars": data.get("stargazers_count", 0),
                "forks": data.get("forks_count", 0),
                "open_issues": data.get("open_issues_count", 0),
                "language": data.get("language", ""),
                "languages": languages,
                "is_template": data.get("is_template", False),
                "is_fork": data.get("fork", False),
                "default_branch": data.get("default_branch", "main"),
                "license": (data.get("license") or {}).get("spdx_id", ""),
                "topics": data.get("topics", []),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "tree_contents": tree_contents,
            }

        except httpx.TimeoutException:
            return {"error": f"Timeout fetching {owner}/{repo} from GitHub.", "source": "github"}
        except Exception as exc:
            await logger.aerror("github_fetch_failed", repo=f"{owner}/{repo}", error=str(exc))
            return {"error": f"Failed to fetch from GitHub: {str(exc)}", "source": "github"}

    async def clone_template(
        self,
        owner: str,
        repo: str,
        target_dir: str,
        new_name: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> dict[str, Any]:
        """Clone a GitHub repo into the project directory as a new project.

        Args:
            owner: GitHub username or org
            repo: Repository name
            target_dir: Parent directory where the project will be created
            new_name: Optional name for the new project folder (defaults to repo name)
            branch: Optional branch to clone (defaults to default branch)

        Returns:
            Status dict with project path and details.
        """
        import pathlib

        project_name = new_name or repo
        target_path = pathlib.Path(target_dir) / project_name

        if target_path.exists():
            return {
                "error": f"Directory '{project_name}' already exists in {target_dir}",
                "source": "github",
            }

        clone_url = f"https://github.com/{owner}/{repo}.git"
        if self.token:
            # Use token for private repos
            clone_url = f"https://{self.token}@github.com/{owner}/{repo}.git"

        try:
            cmd = ["git", "clone", "--depth", "1"]
            if branch:
                cmd.extend(["--branch", branch])
            cmd.extend([clone_url, str(target_path)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=target_dir,
            )

            if result.returncode != 0:
                return {
                    "error": f"Git clone failed: {result.stderr.strip()[:200]}",
                    "source": "github",
                }

            # Remove .git directory to start fresh
            git_dir = target_path / ".git"
            if git_dir.exists():
                shutil.rmtree(git_dir)

            # Re-initialize git
            subprocess.run(
                ["git", "init"],
                capture_output=True,
                cwd=str(target_path),
                timeout=10,
            )

            # Count files
            file_count = sum(1 for _ in target_path.rglob("*") if _.is_file())

            return {
                "source": "github",
                "status": "cloned",
                "project_path": str(target_path),
                "project_name": project_name,
                "from_repo": f"{owner}/{repo}",
                "branch": branch or "default",
                "file_count": file_count,
            }

        except subprocess.TimeoutExpired:
            return {"error": "Git clone timed out after 60 seconds.", "source": "github"}
        except Exception as exc:
            await logger.aerror("github_clone_failed", repo=f"{owner}/{repo}", error=str(exc))
            return {"error": f"Clone failed: {str(exc)}", "source": "github"}

    def format_for_chat(self, data: dict[str, Any]) -> str:
        """Format GitHub repo data as markdown for chat display."""
        if data.get("error"):
            return f"**GitHub Error:** {data['error']}"

        if data.get("status") == "cloned":
            return (
                f"### Project created from [{data.get('from_repo', '')}]({data.get('url', '')})\n\n"
                f"**Location:** `{data.get('project_path', '')}`\n"
                f"**Files:** {data.get('file_count', 0)}\n\n"
                f"The template has been cloned and git re-initialized. You can now start building!"
            )

        lines = []
        lines.append(f"### [{data.get('name', '')}]({data.get('url', '')})")
        if data.get("description"):
            lines.append(f"_{data['description']}_")
        lines.append("")

        meta = []
        if data.get("stars"):
            meta.append(f"**Stars:** {data['stars']:,}")
        if data.get("forks"):
            meta.append(f"**Forks:** {data['forks']:,}")
        if data.get("language"):
            meta.append(f"**Language:** {data['language']}")
        if data.get("license"):
            meta.append(f"**License:** {data['license']}")
        if data.get("is_template"):
            meta.append("**Template repo**")
        if meta:
            lines.append(" · ".join(meta))

        if data.get("topics"):
            lines.append("**Topics:** " + ", ".join(data["topics"][:10]))

        if data.get("tree_contents"):
            lines.append("\n**Files:**")
            for item in data["tree_contents"][:15]:
                icon = "/" if item["type"] == "dir" else ""
                lines.append(f"  `{item['name']}{icon}`")

        return "\n".join(lines)

    def format_for_agent_context(self, data: dict[str, Any]) -> str:
        """Format repo data as context for agent prompts."""
        if data.get("error"):
            return ""

        parts = [
            f"GITHUB REPO: {data.get('name', '')}",
            f"URL: {data.get('url', '')}",
            f"Description: {data.get('description', 'No description')}",
            f"Language: {data.get('language', 'Unknown')} | Stars: {data.get('stars', 0)} | Forks: {data.get('forks', 0)}",
        ]

        if data.get("topics"):
            parts.append(f"Topics: {', '.join(data['topics'])}")

        if data.get("is_template"):
            parts.append("This is a TEMPLATE REPOSITORY — can be cloned as a project starter.")

        if data.get("tree_contents"):
            parts.append("\nTop-level files:")
            for item in data["tree_contents"][:20]:
                icon = "/" if item["type"] == "dir" else ""
                parts.append(f"  {item['name']}{icon}")

        if data.get("languages"):
            langs = ", ".join(f"{k}: {v}" for k, v in list(data["languages"].items())[:8])
            parts.append(f"\nLanguages: {langs}")

        return "\n".join(parts)

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
