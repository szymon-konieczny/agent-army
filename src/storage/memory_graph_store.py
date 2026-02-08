"""In-memory knowledge graph — zero-dependency drop-in for Neo4jStore.

Provides the same async interface as Neo4jStore so the system works out of the
box without Neo4j installed.  Data lives in memory and is lost on restart,
but this is perfectly fine for desktop/development use.
"""

import time
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class InMemoryGraphStore:
    """In-memory graph store matching the Neo4jStore interface.

    Stores nodes as dicts keyed by ``(label, id)`` and relationships as
    adjacency lists.  Supports all high-level operations used by agents:
    knowledge storage/recall, codebase indexing, impact analysis, and
    arbitrary Cypher-like queries (simplified).

    Attributes:
        nodes: Dictionary of ``{node_id: {label, properties}}``
        relationships: List of ``{from_id, to_id, rel_type, properties}``
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.relationships: list[dict[str, Any]] = []
        self._is_connected = False

    # ── Connection lifecycle ──────────────────────────────────────

    async def connect(self) -> None:
        """Mark the store as connected (no-op for in-memory)."""
        self._is_connected = True
        await logger.ainfo("memory_graph_connected")

    async def disconnect(self) -> None:
        """Mark the store as disconnected."""
        self._is_connected = False
        await logger.ainfo("memory_graph_disconnected")

    async def health_check(self) -> bool:
        """Always healthy when connected."""
        return self._is_connected

    def is_connected(self) -> bool:
        """Check connection state."""
        return self._is_connected

    # ── Core graph operations ─────────────────────────────────────

    async def query(
        self,
        cypher: str,
        params: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Simplified query — returns all nodes (Cypher not interpreted)."""
        return list(self.nodes.values())

    async def create_node(
        self,
        label: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a node in the graph."""
        node_id = properties.get("id", str(hash(str(properties))))
        node = {
            "id": node_id,
            "label": label,
            "created_at": time.time(),
            **properties,
        }
        self.nodes[node_id] = node

        await logger.adebug(
            "memory_graph_node_created",
            label=label,
            node_id=node_id,
        )

        return node

    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Create a relationship between two nodes."""
        if from_id not in self.nodes or to_id not in self.nodes:
            return False

        rel = {
            "from_id": from_id,
            "to_id": to_id,
            "rel_type": rel_type,
            "created_at": time.time(),
            **(properties or {}),
        }
        self.relationships.append(rel)

        await logger.adebug(
            "memory_graph_relationship_created",
            rel_type=rel_type,
            from_id=from_id,
            to_id=to_id,
        )

        return True

    async def find_node(
        self,
        label: str,
        properties: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Find a node by label and properties."""
        for node in self.nodes.values():
            if node.get("label") != label:
                continue
            if all(node.get(k) == v for k, v in properties.items()):
                return node
        return None

    async def find_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 5,
    ) -> Optional[list[dict[str, Any]]]:
        """BFS shortest path between two nodes."""
        if from_id not in self.nodes or to_id not in self.nodes:
            return None

        visited: set[str] = set()
        queue: list[list[str]] = [[from_id]]

        while queue:
            path = queue.pop(0)
            current = path[-1]

            if current == to_id:
                return [self.nodes[nid] for nid in path]

            if current in visited or len(path) > max_depth:
                continue
            visited.add(current)

            for rel in self.relationships:
                neighbor = None
                if rel["from_id"] == current:
                    neighbor = rel["to_id"]
                elif rel["to_id"] == current:
                    neighbor = rel["from_id"]
                if neighbor and neighbor not in visited:
                    queue.append(path + [neighbor])

        return None

    async def get_neighbors(
        self,
        node_id: str,
        rel_type: Optional[str] = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Get neighboring nodes."""
        neighbors: list[dict[str, Any]] = []

        for rel in self.relationships:
            if rel_type and rel["rel_type"] != rel_type:
                continue

            neighbor_id = None
            if direction in ("out", "both") and rel["from_id"] == node_id:
                neighbor_id = rel["to_id"]
            if direction in ("in", "both") and rel["to_id"] == node_id:
                neighbor_id = rel["from_id"]

            if neighbor_id and neighbor_id in self.nodes:
                neighbors.append(self.nodes[neighbor_id])

        return neighbors

    # ── Codebase indexing ─────────────────────────────────────────

    async def index_file(
        self,
        filepath: str,
        metadata: dict[str, Any],
    ) -> str:
        """Create a File node."""
        properties = {
            "id": filepath,
            "path": filepath,
            "name": filepath.split("/")[-1],
            **metadata,
        }
        await self.create_node("File", properties)
        return filepath

    async def index_function(
        self,
        name: str,
        filepath: str,
        signature: str,
        docstring: str = "",
    ) -> str:
        """Create a Function node linked to a File."""
        func_id = f"{filepath}::{name}"
        properties = {
            "id": func_id,
            "name": name,
            "signature": signature,
            "docstring": docstring,
        }
        await self.create_node("Function", properties)
        await self.create_relationship(func_id, filepath, "DEFINED_IN")
        return func_id

    async def link_dependency(
        self,
        from_file: str,
        to_file: str,
        dep_type: str,
    ) -> bool:
        """Create a dependency relationship between files."""
        return await self.create_relationship(from_file, to_file, dep_type)

    async def link_commit(
        self,
        commit_hash: str,
        files_changed: list[str],
        message: str,
    ) -> bool:
        """Create a Commit node linked to changed files."""
        properties = {
            "id": commit_hash,
            "hash": commit_hash,
            "message": message,
            "file_count": len(files_changed),
        }
        await self.create_node("Commit", properties)
        for fp in files_changed:
            await self.create_relationship(commit_hash, fp, "MODIFIES")
        return True

    async def get_file_dependencies(
        self,
        filepath: str,
    ) -> list[dict[str, Any]]:
        """Get all dependencies for a file."""
        deps: list[dict[str, Any]] = []
        for rel in self.relationships:
            if rel["from_id"] == filepath and rel["rel_type"] in (
                "IMPORTS",
                "CALLS",
                "REFERENCES",
            ):
                target = self.nodes.get(rel["to_id"])
                if target:
                    deps.append({"target": target, "rel_type": rel["rel_type"]})
        return deps

    async def get_impact_analysis(
        self,
        filepath: str,
    ) -> dict[str, Any]:
        """Analyze what breaks if a file changes."""
        impacted_files: list[dict[str, Any]] = []
        for rel in self.relationships:
            if rel["to_id"] == filepath:
                dep = self.nodes.get(rel["from_id"])
                if dep and dep.get("label") == "File":
                    impacted_files.append(
                        {"file": dep, "rel_type": rel["rel_type"]}
                    )

        impacted_functions: list[dict[str, Any]] = []
        for rel in self.relationships:
            if (
                rel["to_id"] == filepath
                and rel["rel_type"] == "DEFINED_IN"
            ):
                func = self.nodes.get(rel["from_id"])
                if func:
                    impacted_functions.append(func)

        return {
            "filepath": filepath,
            "impacted_files": impacted_files,
            "impacted_functions": impacted_functions,
            "severity": "high" if impacted_files else "low",
        }

    # ── Agent knowledge ───────────────────────────────────────────

    async def store_agent_knowledge(
        self,
        agent_id: str,
        topic: str,
        content: str,
        embedding: Optional[list[float]] = None,
    ) -> str:
        """Store agent knowledge in the graph."""
        knowledge_id = f"{agent_id}::{topic}::{hash(content)}"
        properties: dict[str, Any] = {
            "id": knowledge_id,
            "topic": topic,
            "content": content,
            "agent_id": agent_id,
        }
        if embedding:
            properties["embedding"] = embedding

        await self.create_node("Knowledge", properties)

        # Ensure agent node exists
        if agent_id not in self.nodes:
            await self.create_node("Agent", {"id": agent_id, "name": agent_id})

        await self.create_relationship(agent_id, knowledge_id, "KNOWS")

        return knowledge_id

    async def recall_agent_knowledge(
        self,
        agent_id: str,
        topic: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve agent's stored knowledge."""
        results: list[dict[str, Any]] = []

        for rel in self.relationships:
            if rel["from_id"] == agent_id and rel["rel_type"] == "KNOWS":
                node = self.nodes.get(rel["to_id"])
                if not node:
                    continue
                if topic and node.get("topic") != topic:
                    continue
                results.append(node)
                if len(results) >= limit:
                    break

        # Sort newest first
        results.sort(key=lambda n: n.get("created_at", 0), reverse=True)
        return results[:limit]

    async def store_task_result(
        self,
        task_id: str,
        agent_id: str,
        result_summary: str,
    ) -> bool:
        """Store a task result and link to agent."""
        properties = {
            "id": task_id,
            "task_id": task_id,
            "summary": result_summary,
            "agent_id": agent_id,
        }
        await self.create_node("TaskResult", properties)

        if agent_id not in self.nodes:
            await self.create_node("Agent", {"id": agent_id, "name": agent_id})

        await self.create_relationship(agent_id, task_id, "EXECUTED")
        return True
