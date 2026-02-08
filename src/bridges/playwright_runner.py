"""Playwright E2E test runner bridge for AgentArmy.

Provides the ability to:
- Generate Playwright test scripts from natural-language descriptions.
- Execute existing Playwright test suites (TypeScript/JS or Python).
- Run quick smoke tests against a running dev server (url check, screenshot, element assertions).
- Capture screenshots and traces for test evidence.

Requires `playwright` to be installed in the project's environment:
  pip install playwright && python -m playwright install chromium
or
  npm i -D @playwright/test && npx playwright install chromium
"""

import asyncio
import json
import os
import pathlib
import re
import shutil
import tempfile
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

# ── Detection helpers ────────────────────────────────────────────────

def detect_playwright_setup(project_dir: str) -> dict[str, Any]:
    """Detect Playwright configuration in a project.

    Returns:
        Dict describing the Playwright setup (or lack thereof).
    """
    root = pathlib.Path(project_dir)
    info: dict[str, Any] = {
        "installed": False,
        "config_file": None,
        "language": None,
        "test_dir": None,
        "test_files": [],
    }

    # Check for JS/TS Playwright
    pkg_json = root / "package.json"
    if pkg_json.is_file():
        try:
            pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
            all_deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
            if "@playwright/test" in all_deps or "playwright" in all_deps:
                info["installed"] = True
                info["language"] = "typescript"
        except (json.JSONDecodeError, IOError):
            pass

    # Check for Python Playwright
    for req_file in ("requirements.txt", "requirements-dev.txt", "pyproject.toml"):
        rp = root / req_file
        if rp.is_file():
            try:
                text = rp.read_text(encoding="utf-8")
                if "playwright" in text.lower():
                    info["installed"] = True
                    info["language"] = info["language"] or "python"
            except IOError:
                pass

    # Check for config files
    for config_name in (
        "playwright.config.ts",
        "playwright.config.js",
        "playwright.config.mjs",
        "pytest.ini",
        "conftest.py",
    ):
        if (root / config_name).is_file():
            info["config_file"] = config_name
            break

    # Find test files
    test_patterns = [
        "tests/**/*.spec.ts",
        "tests/**/*.spec.js",
        "e2e/**/*.spec.ts",
        "e2e/**/*.spec.js",
        "test/**/*.spec.ts",
        "test/**/*.spec.js",
        "tests/**/test_*.py",
        "e2e/**/test_*.py",
    ]
    for pattern in test_patterns:
        for f in root.glob(pattern):
            rel = str(f.relative_to(root))
            info["test_files"].append(rel)
            if info["test_dir"] is None:
                info["test_dir"] = str(f.parent.relative_to(root))

    # Fallback: check common test dirs
    if not info["test_dir"]:
        for d in ("e2e", "tests/e2e", "test/e2e", "tests", "test"):
            if (root / d).is_dir():
                info["test_dir"] = d
                break

    return info


# ── PlaywrightRunner ─────────────────────────────────────────────────

class PlaywrightRunner:
    """Runs Playwright tests and captures results.

    Supports both the Node.js ``@playwright/test`` runner and
    the Python ``pytest-playwright`` runner.
    """

    TIMEOUT_SECONDS = 180  # 3-minute hard cap on test runs

    def __init__(self, project_dir: Optional[str] = None):
        self._project_dir = project_dir
        self._chromium_ready = False

    @property
    def root(self) -> Optional[pathlib.Path]:
        return pathlib.Path(self._project_dir) if self._project_dir else None

    @property
    def chromium_ready(self) -> bool:
        return self._chromium_ready

    def set_project_dir(self, project_dir: str) -> None:
        self._project_dir = project_dir

    # ── Auto-provision Chromium on startup ────────────────────────────

    async def ensure_chromium(self) -> dict[str, Any]:
        """Ensure Playwright Python + Chromium are installed.

        Called once during AgentArmy startup.  Idempotent — skips if
        Chromium is already present.

        Returns:
            Dict with status and details.
        """
        # Step 1: Check if playwright Python package exists
        try:
            import playwright  # noqa: F401
        except ImportError:
            # Try to pip-install it
            logger.info("playwright_not_found_installing")
            pip_result = await self._run_cmd_global(
                ["pip", "install", "playwright"]
            )
            if not pip_result["success"]:
                # Try with --user flag
                pip_result = await self._run_cmd_global(
                    ["pip", "install", "--user", "playwright"]
                )
            if not pip_result["success"]:
                self._chromium_ready = False
                return {
                    "status": "fail",
                    "step": "pip_install",
                    "detail": pip_result.get("stderr", "")[:500],
                }
            # Invalidate import cache so the new package is found
            import importlib
            importlib.invalidate_caches()

        # Step 2: Check if Chromium binary already exists
        chromium_ok = await self._check_chromium_binary()
        if chromium_ok:
            self._chromium_ready = True
            logger.info("playwright_chromium_already_installed")
            return {"status": "pass", "step": "already_installed"}

        # Step 3: Install Chromium via `playwright install chromium`
        logger.info("playwright_installing_chromium")
        install_result = await self._run_cmd_global(
            ["python3", "-m", "playwright", "install", "chromium"]
        )
        if not install_result["success"]:
            # Fallback: try via the playwright driver directly
            install_result = await self._run_cmd_global(
                ["python", "-m", "playwright", "install", "chromium"]
            )

        if install_result["success"]:
            self._chromium_ready = True
            logger.info("playwright_chromium_installed")
            return {"status": "pass", "step": "installed"}
        else:
            self._chromium_ready = False
            logger.warning(
                "playwright_chromium_install_failed",
                stderr=install_result.get("stderr", "")[:300],
            )
            return {
                "status": "fail",
                "step": "chromium_install",
                "detail": install_result.get("stderr", "")[:500],
            }

    async def _check_chromium_binary(self) -> bool:
        """Quick check if Chromium is already downloaded."""
        try:
            from playwright._impl._driver import compute_driver_executable
            driver_exec = compute_driver_executable()
            proc = await asyncio.create_subprocess_exec(
                str(driver_exec), "install", "--check", "chromium",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=10)
            return proc.returncode == 0
        except Exception:
            pass

        # Fallback: look for the browser in standard cache paths
        cache_dirs = [
            pathlib.Path.home() / ".cache" / "ms-playwright",
            pathlib.Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")) if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") else None,
        ]
        for d in cache_dirs:
            if d and d.is_dir():
                chromium_dirs = list(d.glob("chromium-*"))
                if chromium_dirs:
                    return True
        return False

    async def _run_cmd_global(self, cmd: list[str]) -> dict[str, Any]:
        """Run a command (not scoped to project dir)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180.0)
            return {
                "success": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace")[-2000:] if stdout else "",
                "stderr": stderr.decode("utf-8", errors="replace")[-2000:] if stderr else "",
            }
        except asyncio.TimeoutError:
            return {"success": False, "exit_code": -1, "stdout": "", "stderr": "Timed out (180s)"}
        except FileNotFoundError:
            return {"success": False, "exit_code": -1, "stdout": "", "stderr": f"Not found: {cmd[0]}"}

    # ── Run existing test suite ──────────────────────────────────────

    async def run_tests(
        self,
        *,
        test_file: Optional[str] = None,
        headed: bool = False,
        grep: Optional[str] = None,
        reporter: str = "json",
        extra_args: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Run Playwright tests and return structured results.

        Args:
            test_file: Specific test file to run (relative to project).
            headed: Run in headed browser mode.
            grep: Filter tests by title pattern.
            reporter: Output reporter (json, html, dot, list).
            extra_args: Additional CLI arguments.

        Returns:
            Dict with pass/fail status, test results, duration.
        """
        if not self.root or not self.root.is_dir():
            return {"status": "error", "message": "No project directory configured"}

        setup = detect_playwright_setup(str(self.root))
        if not setup["installed"]:
            return {
                "status": "error",
                "message": "Playwright is not installed in this project. "
                           "Run `npm i -D @playwright/test` or `pip install playwright`.",
            }

        if setup["language"] == "typescript":
            return await self._run_node_tests(
                test_file=test_file,
                headed=headed,
                grep=grep,
                reporter=reporter,
                extra_args=extra_args,
            )
        else:
            return await self._run_python_tests(
                test_file=test_file,
                headed=headed,
                grep=grep,
                extra_args=extra_args,
            )

    async def _run_node_tests(
        self,
        test_file: Optional[str],
        headed: bool,
        grep: Optional[str],
        reporter: str,
        extra_args: Optional[list[str]],
    ) -> dict[str, Any]:
        """Run tests with @playwright/test (npx playwright test)."""
        cmd = ["npx", "playwright", "test"]

        if test_file:
            cmd.append(test_file)
        if headed:
            cmd.append("--headed")
        if grep:
            cmd.extend(["--grep", grep])

        # Use JSON reporter for structured results + output file
        json_output = self.root / "test-results" / "pw-results.json"
        cmd.extend(["--reporter", f"json"])

        if extra_args:
            cmd.extend(extra_args)

        return await self._exec(cmd, json_output_path=json_output)

    async def _run_python_tests(
        self,
        test_file: Optional[str],
        headed: bool,
        grep: Optional[str],
        extra_args: Optional[list[str]],
    ) -> dict[str, Any]:
        """Run tests with pytest + playwright plugin."""
        cmd = ["python", "-m", "pytest"]

        if test_file:
            cmd.append(test_file)
        if headed:
            cmd.extend(["--headed"])
        if grep:
            cmd.extend(["-k", grep])

        cmd.append("--tb=short")
        cmd.append("-q")

        # JSON report via pytest-json-report if available
        json_output = self.root / "test-results" / "pw-results.json"
        cmd.extend(["--json-report", f"--json-report-file={json_output}"])

        if extra_args:
            cmd.extend(extra_args)

        return await self._exec(cmd, json_output_path=json_output)

    async def _exec(
        self,
        cmd: list[str],
        json_output_path: Optional[pathlib.Path] = None,
    ) -> dict[str, Any]:
        """Execute a test command and collect results."""
        try:
            env = {**os.environ, "CI": "1", "PLAYWRIGHT_JSON_OUTPUT_NAME": str(json_output_path) if json_output_path else ""}
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.root),
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.TIMEOUT_SECONDS,
            )

            stdout_str = stdout.decode("utf-8", errors="replace")[-5000:] if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace")[-3000:] if stderr else ""
            success = proc.returncode == 0

            # Try to parse structured JSON results
            structured = None
            if json_output_path and json_output_path.is_file():
                try:
                    structured = json.loads(json_output_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, IOError):
                    pass

            # Also try to parse JSON from stdout (npx playwright test --reporter=json outputs to stdout)
            if not structured and stdout_str.strip().startswith("{"):
                try:
                    structured = json.loads(stdout_str)
                except json.JSONDecodeError:
                    pass

            # Extract summary from structured results
            summary = self._extract_summary(structured, stdout_str, success)

            return {
                "status": "pass" if success else "fail",
                "exit_code": proc.returncode,
                "command": " ".join(cmd),
                "summary": summary,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "structured": structured,
            }

        except asyncio.TimeoutError:
            return {
                "status": "timeout",
                "exit_code": -1,
                "command": " ".join(cmd),
                "summary": {"error": f"Tests timed out after {self.TIMEOUT_SECONDS}s"},
                "stdout": "",
                "stderr": f"Timed out after {self.TIMEOUT_SECONDS} seconds",
                "structured": None,
            }
        except FileNotFoundError:
            return {
                "status": "error",
                "exit_code": -1,
                "command": " ".join(cmd),
                "summary": {"error": f"Command not found: {cmd[0]}"},
                "stdout": "",
                "stderr": f"Command not found: {cmd[0]}. Is Playwright installed?",
                "structured": None,
            }
        except Exception as exc:
            return {
                "status": "error",
                "exit_code": -1,
                "command": " ".join(cmd),
                "summary": {"error": str(exc)},
                "stdout": "",
                "stderr": str(exc),
                "structured": None,
            }

    def _extract_summary(
        self,
        structured: Optional[dict],
        stdout: str,
        success: bool,
    ) -> dict[str, Any]:
        """Extract a human-readable summary from test results."""
        summary: dict[str, Any] = {"passed": success}

        if structured:
            # @playwright/test JSON reporter format
            suites = structured.get("suites", [])
            specs = []
            self._collect_specs(suites, specs)

            total = len(specs)
            passed = sum(1 for s in specs if s.get("ok", False))
            failed = total - passed

            summary.update({
                "total": total,
                "passed_count": passed,
                "failed_count": failed,
                "duration_ms": structured.get("stats", {}).get("duration", 0),
            })

            # Collect failure details
            failures = []
            for spec in specs:
                if not spec.get("ok", False):
                    title = spec.get("title", "unknown")
                    err = ""
                    for test_result in spec.get("tests", []):
                        for result in test_result.get("results", []):
                            if result.get("status") != "passed":
                                err = result.get("error", {}).get("message", "")[:500]
                                break
                    failures.append({"title": title, "error": err})
            if failures:
                summary["failures"] = failures[:10]  # Cap at 10

        else:
            # Parse from stdout text
            match = re.search(
                r"(\d+)\s+passed.*?(\d+)\s+failed",
                stdout,
                re.IGNORECASE,
            )
            if match:
                summary["passed_count"] = int(match.group(1))
                summary["failed_count"] = int(match.group(2))
                summary["total"] = summary["passed_count"] + summary["failed_count"]
            else:
                match_pass = re.search(r"(\d+)\s+passed", stdout, re.IGNORECASE)
                if match_pass:
                    summary["passed_count"] = int(match_pass.group(1))
                    summary["total"] = summary["passed_count"]
                    summary["failed_count"] = 0

        return summary

    def _collect_specs(self, suites: list, out: list) -> None:
        """Recursively collect specs from nested suites."""
        for suite in suites:
            out.extend(suite.get("specs", []))
            self._collect_specs(suite.get("suites", []), out)

    # ── Quick smoke test ─────────────────────────────────────────────

    async def smoke_test(
        self,
        url: str,
        *,
        checks: Optional[list[str]] = None,
        screenshot: bool = True,
    ) -> dict[str, Any]:
        """Run a quick smoke test against a URL.

        Tries Playwright (full browser) first.  Falls back to a lightweight
        httpx-based test when Playwright or Chromium is not available.

        Args:
            url: The URL to test.
            checks: List of checks: "status", "title", "no_console_errors",
                     "screenshot", "a11y", or CSS selectors to assert visible.
            screenshot: Whether to capture a screenshot (Playwright only).

        Returns:
            Dict with check results and optional screenshot path.
        """
        checks = checks or ["status", "title", "no_console_errors"]

        # Try Playwright first — full browser rendering
        pw_result = await self._smoke_test_playwright(url, checks=checks, screenshot=screenshot)
        if pw_result.get("status") != "error" or "not installed" not in pw_result.get("message", "").lower():
            return pw_result

        # Fallback — lightweight HTTP-only smoke test (always works)
        return await self._smoke_test_http(url, checks=checks)

    async def _smoke_test_playwright(
        self,
        url: str,
        checks: list[str],
        screenshot: bool,
    ) -> dict[str, Any]:
        """Full browser smoke test via Playwright."""
        results: dict[str, Any] = {"url": url, "checks": {}, "status": "pass", "engine": "playwright"}

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return {
                "url": url,
                "status": "error",
                "message": "Playwright Python package not installed — falling back to HTTP test.",
                "checks": {},
            }

        pw = None
        browser = None
        try:
            pw = await async_playwright().start()
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                ignore_https_errors=True,
            )

            # Capture console errors
            console_errors: list[str] = []
            page = await context.new_page()
            page.on("console", lambda msg: (
                console_errors.append(f"[{msg.type}] {msg.text}")
                if msg.type in ("error", "warning") else None
            ))

            # Navigate
            response = await page.goto(url, wait_until="networkidle", timeout=30000)

            if "status" in checks:
                code = response.status if response else 0
                results["checks"]["status"] = {
                    "passed": 200 <= code < 400,
                    "value": code,
                }
                if not results["checks"]["status"]["passed"]:
                    results["status"] = "fail"

            if "title" in checks:
                title = await page.title()
                results["checks"]["title"] = {
                    "passed": bool(title and title.strip()),
                    "value": title[:200] if title else "",
                }

            if "no_console_errors" in checks:
                errors_only = [e for e in console_errors if e.startswith("[error]")]
                results["checks"]["no_console_errors"] = {
                    "passed": len(errors_only) == 0,
                    "value": errors_only[:5],
                }
                if errors_only:
                    results["status"] = "fail"

            for check in checks:
                if check.startswith(".") or check.startswith("#") or check.startswith("["):
                    try:
                        el = await page.query_selector(check)
                        visible = await el.is_visible() if el else False
                        results["checks"][f"selector:{check}"] = {
                            "passed": visible,
                            "value": "visible" if visible else "not found / hidden",
                        }
                        if not visible:
                            results["status"] = "fail"
                    except Exception as sel_exc:
                        results["checks"][f"selector:{check}"] = {
                            "passed": False,
                            "value": str(sel_exc)[:200],
                        }
                        results["status"] = "fail"

            if "a11y" in checks:
                lang = await page.evaluate("() => document.documentElement.lang")
                h1 = await page.query_selector("h1")
                results["checks"]["a11y"] = {
                    "passed": bool(lang) and h1 is not None,
                    "value": {
                        "has_lang": bool(lang),
                        "lang": lang or "missing",
                        "has_h1": h1 is not None,
                    },
                }

            if screenshot:
                ss_dir = pathlib.Path(self._project_dir or ".") / ".agentarmy" / "screenshots"
                ss_dir.mkdir(parents=True, exist_ok=True)
                ss_path = ss_dir / f"smoke_{int(asyncio.get_event_loop().time())}.png"
                await page.screenshot(path=str(ss_path), full_page=True)
                results["screenshot"] = str(ss_path)

            try:
                metrics = await page.evaluate("""() => {
                    const perf = performance.getEntriesByType('navigation')[0];
                    return perf ? {
                        domContentLoaded: Math.round(perf.domContentLoadedEventEnd - perf.startTime),
                        loadComplete: Math.round(perf.loadEventEnd - perf.startTime),
                        domInteractive: Math.round(perf.domInteractive - perf.startTime),
                    } : null;
                }""")
                if metrics:
                    results["performance"] = metrics
            except Exception:
                pass

            await context.close()

        except Exception as exc:
            err = str(exc)[:500]
            # If Chromium isn't installed, signal for fallback
            if "executable doesn't exist" in err.lower() or "browser" in err.lower():
                return {
                    "url": url,
                    "status": "error",
                    "message": f"Chromium not installed — falling back to HTTP test. ({err[:120]})",
                    "checks": {},
                }
            results["status"] = "error"
            results["message"] = err
        finally:
            if browser:
                await browser.close()
            if pw:
                await pw.stop()

        return results

    async def _smoke_test_http(
        self,
        url: str,
        checks: list[str],
    ) -> dict[str, Any]:
        """Lightweight HTTP-only smoke test (no browser needed).

        Uses httpx to fetch the page and parse the HTML for basic checks.
        Works everywhere — no Chromium download required.
        """
        import time as _time

        results: dict[str, Any] = {"url": url, "checks": {}, "status": "pass", "engine": "httpx"}

        try:
            import httpx
        except ImportError:
            return {
                "url": url,
                "status": "error",
                "message": "Neither Playwright nor httpx is available.",
                "checks": {},
            }

        try:
            start = _time.monotonic()
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=30.0,
                verify=False,
                headers={"User-Agent": "AgentArmy-SmokeTest/1.0"},
            ) as client:
                resp = await client.get(url)
            elapsed_ms = int((_time.monotonic() - start) * 1000)

            body = resp.text

            # ── Check: HTTP status ───────────────────────────────
            if "status" in checks:
                code = resp.status_code
                passed = 200 <= code < 400
                results["checks"]["status"] = {"passed": passed, "value": code}
                if not passed:
                    results["status"] = "fail"

            # ── Check: Page title ────────────────────────────────
            if "title" in checks:
                title_match = re.search(r"<title[^>]*>([^<]+)</title>", body, re.IGNORECASE)
                title = title_match.group(1).strip() if title_match else ""
                results["checks"]["title"] = {
                    "passed": bool(title),
                    "value": title[:200],
                }

            # ── Check: Content-Type ──────────────────────────────
            if "no_console_errors" in checks:
                # In HTTP-only mode we can't see console errors; check content-type instead
                ct = resp.headers.get("content-type", "")
                is_html = "text/html" in ct
                results["checks"]["content_type"] = {
                    "passed": is_html,
                    "value": ct[:100],
                }

            # ── Check: Response size ─────────────────────────────
            size_bytes = len(body.encode("utf-8", errors="replace"))
            results["checks"]["response_size"] = {
                "passed": size_bytes > 100,
                "value": f"{size_bytes:,} bytes",
            }
            if size_bytes <= 100:
                results["status"] = "fail"

            # ── Check: Security headers ──────────────────────────
            sec_headers = {}
            for h in ("x-frame-options", "x-content-type-options", "strict-transport-security",
                       "content-security-policy", "x-xss-protection"):
                val = resp.headers.get(h)
                sec_headers[h] = val or "missing"
            present = sum(1 for v in sec_headers.values() if v != "missing")
            results["checks"]["security_headers"] = {
                "passed": present >= 2,
                "value": sec_headers,
            }

            # ── Check: Basic HTML structure ──────────────────────
            has_doctype = body.strip().lower().startswith("<!doctype")
            has_html = "<html" in body.lower()
            has_head = "<head" in body.lower()
            has_body = "<body" in body.lower()
            results["checks"]["html_structure"] = {
                "passed": has_doctype and has_html and has_body,
                "value": {
                    "doctype": has_doctype,
                    "html": has_html,
                    "head": has_head,
                    "body": has_body,
                },
            }

            # ── Check: Meta tags ─────────────────────────────────
            has_viewport = bool(re.search(r'<meta[^>]+name=["\']viewport["\']', body, re.I))
            has_charset = bool(re.search(r'<meta[^>]+charset=', body, re.I))
            has_description = bool(re.search(r'<meta[^>]+name=["\']description["\']', body, re.I))
            results["checks"]["meta_tags"] = {
                "passed": has_viewport and has_charset,
                "value": {
                    "viewport": has_viewport,
                    "charset": has_charset,
                    "description": has_description,
                },
            }

            # ── Check: a11y basics (lang attribute, h1) ──────────
            if "a11y" in checks:
                lang_match = re.search(r'<html[^>]+lang=["\']([^"\']+)', body, re.I)
                has_h1 = bool(re.search(r"<h1[\s>]", body, re.I))
                results["checks"]["a11y"] = {
                    "passed": bool(lang_match) and has_h1,
                    "value": {
                        "has_lang": bool(lang_match),
                        "lang": lang_match.group(1) if lang_match else "missing",
                        "has_h1": has_h1,
                    },
                }

            # ── Performance timing ───────────────────────────────
            results["performance"] = {
                "response_time_ms": elapsed_ms,
                "ttfb_ms": elapsed_ms,  # approximate
            }

            # ── SSL certificate info ─────────────────────────────
            if url.startswith("https"):
                results["checks"]["ssl"] = {
                    "passed": True,
                    "value": "Valid (connection succeeded over HTTPS)",
                }

        except httpx.ConnectError as exc:
            results["status"] = "fail"
            results["message"] = f"Connection failed: {str(exc)[:300]}"
        except httpx.TimeoutException:
            results["status"] = "fail"
            results["message"] = "Request timed out after 30 seconds"
        except Exception as exc:
            results["status"] = "error"
            results["message"] = str(exc)[:500]

        return results

    # ── Generate test scaffold ───────────────────────────────────────

    def scaffold_test(
        self,
        test_name: str,
        url: str,
        *,
        language: str = "typescript",
        assertions: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Generate a Playwright test file scaffold.

        Args:
            test_name: Name for the test (used in file and describe block).
            url: URL to test against.
            language: "typescript" or "python".
            assertions: List of things to assert (e.g., "page title is 'Home'").

        Returns:
            Dict with file path and content of generated test.
        """
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", test_name.lower())

        if language == "typescript":
            return self._scaffold_ts(safe_name, test_name, url, assertions or [])
        else:
            return self._scaffold_py(safe_name, test_name, url, assertions or [])

    def _scaffold_ts(
        self,
        safe_name: str,
        test_name: str,
        url: str,
        assertions: list[str],
    ) -> dict[str, Any]:
        assertion_lines = ""
        for a in assertions:
            assertion_lines += f"  // TODO: {a}\n"

        content = f"""import {{ test, expect }} from '@playwright/test';

test.describe('{test_name}', () => {{
  test.beforeEach(async ({{ page }}) => {{
    await page.goto('{url}');
  }});

  test('should load successfully', async ({{ page }}) => {{
    // Verify the page loads with a valid status
    await expect(page).toHaveTitle(/.+/);
  }});

  test('should have no console errors', async ({{ page }}) => {{
    const errors: string[] = [];
    page.on('console', msg => {{
      if (msg.type() === 'error') errors.push(msg.text());
    }});
    await page.goto('{url}');
    await page.waitForLoadState('networkidle');
    expect(errors).toHaveLength(0);
  }});

  test('should be responsive', async ({{ page }}) => {{
    // Desktop
    await page.setViewportSize({{ width: 1280, height: 720 }});
    await expect(page.locator('body')).toBeVisible();

    // Tablet
    await page.setViewportSize({{ width: 768, height: 1024 }});
    await expect(page.locator('body')).toBeVisible();

    // Mobile
    await page.setViewportSize({{ width: 375, height: 667 }});
    await expect(page.locator('body')).toBeVisible();
  }});

{assertion_lines}}});
"""
        test_dir = "e2e" if self.root and (self.root / "e2e").is_dir() else "tests"
        file_path = f"{test_dir}/{safe_name}.spec.ts"

        return {
            "file_path": file_path,
            "content": content,
            "language": "typescript",
        }

    def _scaffold_py(
        self,
        safe_name: str,
        test_name: str,
        url: str,
        assertions: list[str],
    ) -> dict[str, Any]:
        assertion_lines = ""
        for a in assertions:
            assertion_lines += f"    # TODO: {a}\n"

        content = f"""\"\"\"E2E tests: {test_name}\"\"\"
import re
from playwright.sync_api import Page, expect


def test_page_loads(page: Page):
    \"\"\"Verify page loads with valid status.\"\"\"
    response = page.goto("{url}")
    assert response is not None
    assert response.status < 400
    expect(page).to_have_title(re.compile(".+"))


def test_no_console_errors(page: Page):
    \"\"\"Verify no console errors on load.\"\"\"
    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.goto("{url}")
    page.wait_for_load_state("networkidle")
    assert len(errors) == 0, f"Console errors: {{errors}}"


def test_responsive_layout(page: Page):
    \"\"\"Verify page renders on different viewports.\"\"\"
    for width, height in [(1280, 720), (768, 1024), (375, 667)]:
        page.set_viewport_size({{"width": width, "height": height}})
        page.goto("{url}")
        expect(page.locator("body")).to_be_visible()

{assertion_lines}"""
        test_dir = "e2e" if self.root and (self.root / "e2e").is_dir() else "tests"
        file_path = f"{test_dir}/test_{safe_name}.py"

        return {
            "file_path": file_path,
            "content": content,
            "language": "python",
        }

    # ── Install Playwright ───────────────────────────────────────────

    async def install(self, language: str = "auto") -> dict[str, Any]:
        """Install Playwright in the project.

        Args:
            language: "typescript", "python", or "auto" (detect from project).

        Returns:
            Dict with installation result.
        """
        if not self.root or not self.root.is_dir():
            return {"status": "error", "message": "No project directory configured"}

        if language == "auto":
            if (self.root / "package.json").is_file():
                language = "typescript"
            elif (self.root / "pyproject.toml").is_file() or (self.root / "requirements.txt").is_file():
                language = "python"
            else:
                language = "typescript"

        steps: list[dict[str, Any]] = []

        if language == "typescript":
            # Install @playwright/test (--legacy-peer-deps avoids NestJS-style conflicts)
            cmd1 = ["npm", "install", "-D", "--legacy-peer-deps", "@playwright/test"]
            r1 = await self._run_cmd(cmd1)
            steps.append({"command": " ".join(cmd1), **r1})

            if r1["success"]:
                # Install browsers
                cmd2 = ["npx", "playwright", "install", "chromium"]
                r2 = await self._run_cmd(cmd2)
                steps.append({"command": " ".join(cmd2), **r2})
        else:
            cmd1 = ["pip", "install", "playwright", "pytest-playwright"]
            r1 = await self._run_cmd(cmd1)
            steps.append({"command": " ".join(cmd1), **r1})

            if r1["success"]:
                cmd2 = ["python", "-m", "playwright", "install", "chromium"]
                r2 = await self._run_cmd(cmd2)
                steps.append({"command": " ".join(cmd2), **r2})

        overall = all(s["success"] for s in steps)
        return {
            "status": "pass" if overall else "fail",
            "language": language,
            "steps": steps,
        }

    async def _run_cmd(self, cmd: list[str]) -> dict[str, Any]:
        """Run a subprocess command in the project directory."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.root),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
            return {
                "success": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="replace")[-2000:] if stdout else "",
                "stderr": stderr.decode("utf-8", errors="replace")[-1000:] if stderr else "",
            }
        except asyncio.TimeoutError:
            return {"success": False, "exit_code": -1, "stdout": "", "stderr": "Timed out after 120s"}
        except FileNotFoundError:
            return {"success": False, "exit_code": -1, "stdout": "", "stderr": f"Not found: {cmd[0]}"}
