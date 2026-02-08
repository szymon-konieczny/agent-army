"""Neo4j knowledge graph for agent memory, codebase relationships, and GraphRAG."""

from typing import Any, Optional
import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError

logger = structlog.get_logger(__name__)


class Neo4jStore:
    """Async Neo4j knowledge graph store for agent memory and codebase analysis.

    Provides async operations for:
    - Knowledge graph node and relationship management
    - Codebase structure indexing (files, functions, dependencies)
    - Agent memory and knowledge storage
    - GraphRAG semantic search and context retrieval
    - Code impact analysis and dependency traversal

    Attributes:
        uri: Neo4j connection URI (bolt://host:port)
        username: Neo4j username
        password: Neo4j password
        driver: Async Neo4j driver instance
        database: Target database name
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        encryption: bool = True,
    ) -> None:
        """Initialize Neo4j knowledge graph connection.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687").
            username: Neo4j authentication username.
            password: Neo4j authentication password.
            database: Target database name (default: "neo4j").
            encryption: Whether to use encrypted connections (default: True).

        Raises:
            ValueError: If URI, username, or password is empty.
        """
        if not uri or not username or not password:
            raise ValueError("URI, username, and password are required")

        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[AsyncDriver] = None
        self._is_connected = False

    async def connect(self) -> None:
        """Establish async Neo4j connection pool.

        Creates async driver and verifies connectivity.

        Raises:
            Neo4jError: If connection fails.
            Exception: For other connection issues.
        """
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )

            # Test connection
            async with self.driver.session(database=self.database) as session:
                await session.run("RETURN 1")

            self._is_connected = True

            await logger.ainfo(
                "neo4j_connected",
                uri=self.uri,
                database=self.database,
            )

        except Neo4jError as exc:
            await logger.aerror(
                "neo4j_connection_failed",
                error=str(exc),
                uri=self.uri,
            )
            raise
        except Exception as exc:
            await logger.aerror(
                "neo4j_connection_error",
                error=str(exc),
                uri=self.uri,
            )
            raise

    async def disconnect(self) -> None:
        """Close Neo4j connection.

        Raises:
            Exception: If disconnection fails.
        """
        if not self.driver:
            return

        try:
            await self.driver.close()
            self._is_connected = False

            await logger.ainfo("neo4j_disconnected")

        except Exception as exc:
            await logger.aerror(
                "neo4j_disconnection_failed",
                error=str(exc),
            )
            raise

    async def health_check(self) -> bool:
        """Check Neo4j connection health.

        Returns:
            True if connection is healthy, False otherwise.
        """
        if not self.driver or not self._is_connected:
            return False

        try:
            async with self.driver.session(database=self.database) as session:
                await session.run("RETURN 1")
            return True
        except Exception as exc:
            await logger.awarning(
                "neo4j_health_check_failed",
                error=str(exc),
            )
            return False

    async def query(
        self,
        cypher: str,
        params: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results.

        Args:
            cypher: Cypher query string.
            params: Query parameters (default: None).

        Returns:
            List of result records as dictionaries.

        Raises:
            RuntimeError: If not connected.
            Neo4jError: If query execution fails.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(cypher, params or {})
                records = await result.data()

            await logger.adebug(
                "neo4j_query_executed",
                cypher=cypher[:100],
                result_count=len(records),
            )

            return records

        except Neo4jError as exc:
            await logger.aerror(
                "neo4j_query_failed",
                cypher=cypher[:100],
                error=str(exc),
            )
            raise
        except Exception as exc:
            await logger.aerror(
                "neo4j_query_error",
                cypher=cypher[:100],
                error=str(exc),
            )
            raise

    async def create_node(
        self,
        label: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a node in the knowledge graph.

        Args:
            label: Node label (e.g., "File", "Function", "Agent").
            properties: Node properties as dictionary.

        Returns:
            Dictionary containing created node data with id.

        Raises:
            RuntimeError: If not connected.
            Neo4jError: If creation fails.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            # Generate node ID from properties if not provided
            node_id = properties.get("id", str(hash(str(properties))))

            cypher = f"""
            CREATE (n:{label} $props)
            SET n.id = $id
            SET n.created_at = datetime()
            RETURN n
            """

            params = {
                "props": properties,
                "id": node_id,
            }

            records = await self.query(cypher, params)

            if records:
                await logger.adebug(
                    "neo4j_node_created",
                    label=label,
                    node_id=node_id,
                )
                return {"id": node_id, **properties}

            return {}

        except Exception as exc:
            await logger.aerror(
                "neo4j_create_node_failed",
                label=label,
                error=str(exc),
            )
            raise

    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Create a relationship between two nodes.

        Args:
            from_id: ID of source node.
            to_id: ID of target node.
            rel_type: Relationship type (e.g., "IMPORTS", "CALLS").
            properties: Relationship properties (default: None).

        Returns:
            True if relationship created, False otherwise.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            cypher = f"""
            MATCH (from {{id: $from_id}})
            MATCH (to {{id: $to_id}})
            CREATE (from)-[r:{rel_type} $props]->(to)
            SET r.created_at = datetime()
            RETURN r
            """

            params = {
                "from_id": from_id,
                "to_id": to_id,
                "props": properties or {},
            }

            records = await self.query(cypher, params)

            await logger.adebug(
                "neo4j_relationship_created",
                rel_type=rel_type,
                from_id=from_id,
                to_id=to_id,
            )

            return len(records) > 0

        except Exception as exc:
            await logger.aerror(
                "neo4j_create_relationship_failed",
                rel_type=rel_type,
                from_id=from_id,
                to_id=to_id,
                error=str(exc),
            )
            return False

    async def find_node(
        self,
        label: str,
        properties: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Find a node by label and properties.

        Args:
            label: Node label to search for.
            properties: Properties to match.

        Returns:
            Node data if found, None otherwise.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            # Build WHERE clause dynamically
            where_clauses = [f"n.{k} = ${k}" for k in properties.keys()]
            where_str = " AND ".join(where_clauses) if where_clauses else "1=1"

            cypher = f"""
            MATCH (n:{label})
            WHERE {where_str}
            RETURN n
            LIMIT 1
            """

            records = await self.query(cypher, properties)

            if records:
                await logger.adebug(
                    "neo4j_node_found",
                    label=label,
                    properties=properties,
                )
                return records[0].get("n")

            return None

        except Exception as exc:
            await logger.aerror(
                "neo4j_find_node_failed",
                label=label,
                error=str(exc),
            )
            raise

    async def find_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 5,
    ) -> Optional[list[dict[str, Any]]]:
        """Find shortest path between two nodes.

        Args:
            from_id: ID of source node.
            to_id: ID of target node.
            max_depth: Maximum path depth (default: 5).

        Returns:
            List of nodes in shortest path, or None if no path found.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            cypher = f"""
            MATCH path = shortestPath(
                (from {{id: $from_id}})-[*..{max_depth}]-(to {{id: $to_id}})
            )
            RETURN [node in nodes(path) | node] as path
            """

            params = {
                "from_id": from_id,
                "to_id": to_id,
            }

            records = await self.query(cypher, params)

            if records and "path" in records[0]:
                path = records[0]["path"]
                await logger.adebug(
                    "neo4j_path_found",
                    from_id=from_id,
                    to_id=to_id,
                    path_length=len(path),
                )
                return path

            return None

        except Exception as exc:
            await logger.aerror(
                "neo4j_find_path_failed",
                from_id=from_id,
                to_id=to_id,
                error=str(exc),
            )
            raise

    async def get_neighbors(
        self,
        node_id: str,
        rel_type: Optional[str] = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Get neighboring nodes connected by relationships.

        Args:
            node_id: ID of center node.
            rel_type: Filter by relationship type (default: None for all).
            direction: Relationship direction ("in", "out", "both").

        Returns:
            List of neighboring node data.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            # Build relationship pattern based on direction
            if direction == "in":
                rel_pattern = f"<-[{f':{rel_type}' if rel_type else ''}]-"
            elif direction == "out":
                rel_pattern = f"-[{f':{rel_type}' if rel_type else ''}]->"
            else:  # both
                rel_pattern = f"-[{f':{rel_type}' if rel_type else ''}]-"

            cypher = f"""
            MATCH (center {{id: $node_id}}){rel_pattern}(neighbor)
            RETURN neighbor
            """

            params = {"node_id": node_id}

            records = await self.query(cypher, params)

            neighbors = [r.get("neighbor") for r in records if "neighbor" in r]

            await logger.adebug(
                "neo4j_neighbors_retrieved",
                node_id=node_id,
                rel_type=rel_type,
                neighbor_count=len(neighbors),
            )

            return neighbors

        except Exception as exc:
            await logger.aerror(
                "neo4j_get_neighbors_failed",
                node_id=node_id,
                rel_type=rel_type,
                error=str(exc),
            )
            raise

    async def index_file(
        self,
        filepath: str,
        metadata: dict[str, Any],
    ) -> str:
        """Create File node in knowledge graph.

        Args:
            filepath: Full path to file.
            metadata: File metadata (lines, hash, language, etc.).

        Returns:
            ID of created File node.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            properties = {
                "id": filepath,
                "path": filepath,
                "name": filepath.split("/")[-1],
                **metadata,
            }

            await self.create_node("File", properties)

            await logger.adebug(
                "neo4j_file_indexed",
                filepath=filepath,
            )

            return filepath

        except Exception as exc:
            await logger.aerror(
                "neo4j_index_file_failed",
                filepath=filepath,
                error=str(exc),
            )
            raise

    async def index_function(
        self,
        name: str,
        filepath: str,
        signature: str,
        docstring: str = "",
    ) -> str:
        """Create Function node linked to File node.

        Args:
            name: Function name.
            filepath: File containing function.
            signature: Function signature.
            docstring: Function docstring (default: "").

        Returns:
            ID of created Function node.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            func_id = f"{filepath}::{name}"

            properties = {
                "id": func_id,
                "name": name,
                "signature": signature,
                "docstring": docstring,
            }

            await self.create_node("Function", properties)

            # Link to file
            await self.create_relationship(
                from_id=func_id,
                to_id=filepath,
                rel_type="DEFINED_IN",
            )

            await logger.adebug(
                "neo4j_function_indexed",
                name=name,
                filepath=filepath,
            )

            return func_id

        except Exception as exc:
            await logger.aerror(
                "neo4j_index_function_failed",
                name=name,
                filepath=filepath,
                error=str(exc),
            )
            raise

    async def link_dependency(
        self,
        from_file: str,
        to_file: str,
        dep_type: str,
    ) -> bool:
        """Create dependency relationship between files.

        Args:
            from_file: Source file path.
            to_file: Target file path.
            dep_type: Dependency type (IMPORTS, CALLS, REFERENCES).

        Returns:
            True if relationship created, False otherwise.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            return await self.create_relationship(
                from_id=from_file,
                to_id=to_file,
                rel_type=dep_type,
            )

        except Exception as exc:
            await logger.aerror(
                "neo4j_link_dependency_failed",
                from_file=from_file,
                to_file=to_file,
                dep_type=dep_type,
                error=str(exc),
            )
            raise

    async def link_commit(
        self,
        commit_hash: str,
        files_changed: list[str],
        message: str,
    ) -> bool:
        """Create Commit node and link to changed files.

        Args:
            commit_hash: Git commit hash.
            files_changed: List of file paths modified.
            message: Commit message.

        Returns:
            True if commit linked successfully, False otherwise.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            properties = {
                "id": commit_hash,
                "hash": commit_hash,
                "message": message,
                "file_count": len(files_changed),
            }

            await self.create_node("Commit", properties)

            # Link to each modified file
            for file_path in files_changed:
                await self.create_relationship(
                    from_id=commit_hash,
                    to_id=file_path,
                    rel_type="MODIFIES",
                )

            await logger.adebug(
                "neo4j_commit_linked",
                commit_hash=commit_hash,
                file_count=len(files_changed),
            )

            return True

        except Exception as exc:
            await logger.aerror(
                "neo4j_link_commit_failed",
                commit_hash=commit_hash,
                error=str(exc),
            )
            raise

    async def get_file_dependencies(
        self,
        filepath: str,
    ) -> list[dict[str, Any]]:
        """Get all dependencies for a file.

        Args:
            filepath: Path to target file.

        Returns:
            List of dependent file information.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            cypher = """
            MATCH (from:File {id: $filepath})-[rel]->(to:File)
            RETURN to, type(rel) as rel_type
            """

            params = {"filepath": filepath}

            records = await self.query(cypher, params)

            deps = [
                {
                    "target": r.get("to"),
                    "rel_type": r.get("rel_type"),
                }
                for r in records
                if "to" in r
            ]

            await logger.adebug(
                "neo4j_file_dependencies_retrieved",
                filepath=filepath,
                dep_count=len(deps),
            )

            return deps

        except Exception as exc:
            await logger.aerror(
                "neo4j_get_file_dependencies_failed",
                filepath=filepath,
                error=str(exc),
            )
            raise

    async def get_impact_analysis(
        self,
        filepath: str,
    ) -> dict[str, Any]:
        """Analyze what breaks if a file changes.

        Args:
            filepath: Path to target file.

        Returns:
            Dictionary with impacted files and functions.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            # Find files that depend on this file
            cypher = """
            MATCH (target:File {id: $filepath})<-[rel]-(dependent:File)
            RETURN dependent, type(rel) as rel_type
            """

            params = {"filepath": filepath}

            records = await self.query(cypher, params)

            impacted_files = [
                {
                    "file": r.get("dependent"),
                    "rel_type": r.get("rel_type"),
                }
                for r in records
                if "dependent" in r
            ]

            # Find functions affected through these files
            func_cypher = """
            MATCH (target:File {id: $filepath})<-[:DEFINED_IN]-(func:Function)
            RETURN func
            """

            func_records = await self.query(func_cypher, params)

            impacted_functions = [r.get("func") for r in func_records if "func" in r]

            result = {
                "filepath": filepath,
                "impacted_files": impacted_files,
                "impacted_functions": impacted_functions,
                "severity": "high" if impacted_files else "low",
            }

            await logger.adebug(
                "neo4j_impact_analysis_completed",
                filepath=filepath,
                impacted_count=len(impacted_files),
            )

            return result

        except Exception as exc:
            await logger.aerror(
                "neo4j_get_impact_analysis_failed",
                filepath=filepath,
                error=str(exc),
            )
            raise

    async def store_agent_knowledge(
        self,
        agent_id: str,
        topic: str,
        content: str,
        embedding: Optional[list[float]] = None,
    ) -> str:
        """Store agent knowledge in the graph.

        Args:
            agent_id: ID of agent storing knowledge.
            topic: Knowledge topic/category.
            content: Knowledge content.
            embedding: Vector embedding (optional).

        Returns:
            ID of created knowledge node.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            knowledge_id = f"{agent_id}::{topic}::{hash(content)}"

            properties = {
                "id": knowledge_id,
                "topic": topic,
                "content": content,
                "agent_id": agent_id,
            }

            if embedding:
                properties["embedding"] = embedding

            await self.create_node("Knowledge", properties)

            # Link to agent
            await self.create_relationship(
                from_id=agent_id,
                to_id=knowledge_id,
                rel_type="KNOWS",
            )

            await logger.adebug(
                "neo4j_agent_knowledge_stored",
                agent_id=agent_id,
                topic=topic,
            )

            return knowledge_id

        except Exception as exc:
            await logger.aerror(
                "neo4j_store_agent_knowledge_failed",
                agent_id=agent_id,
                topic=topic,
                error=str(exc),
            )
            raise

    async def recall_agent_knowledge(
        self,
        agent_id: str,
        topic: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve agent's stored knowledge.

        Args:
            agent_id: ID of agent.
            topic: Filter by topic (optional).
            limit: Maximum results to return.

        Returns:
            List of knowledge records.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            topic_filter = f"AND k.topic = '{topic}'" if topic else ""

            cypher = f"""
            MATCH (agent {{id: $agent_id}})-[:KNOWS]->(k:Knowledge)
            WHERE 1=1 {topic_filter}
            RETURN k
            ORDER BY k.created_at DESC
            LIMIT $limit
            """

            params = {
                "agent_id": agent_id,
                "limit": limit,
            }

            records = await self.query(cypher, params)

            knowledge = [r.get("k") for r in records if "k" in r]

            await logger.adebug(
                "neo4j_agent_knowledge_recalled",
                agent_id=agent_id,
                topic=topic,
                count=len(knowledge),
            )

            return knowledge

        except Exception as exc:
            await logger.aerror(
                "neo4j_recall_agent_knowledge_failed",
                agent_id=agent_id,
                topic=topic,
                error=str(exc),
            )
            raise

    async def store_task_result(
        self,
        task_id: str,
        agent_id: str,
        result_summary: str,
    ) -> bool:
        """Store task result and link to agent knowledge.

        Args:
            task_id: ID of completed task.
            agent_id: ID of executing agent.
            result_summary: Summary of task result.

        Returns:
            True if stored successfully, False otherwise.

        Raises:
            RuntimeError: If not connected.
        """
        if not self.driver:
            raise RuntimeError("Neo4j not connected. Call connect() first.")

        try:
            properties = {
                "id": task_id,
                "task_id": task_id,
                "summary": result_summary,
                "agent_id": agent_id,
            }

            await self.create_node("TaskResult", properties)

            # Link to agent
            await self.create_relationship(
                from_id=agent_id,
                to_id=task_id,
                rel_type="EXECUTED",
            )

            await logger.adebug(
                "neo4j_task_result_stored",
                task_id=task_id,
                agent_id=agent_id,
            )

            return True

        except Exception as exc:
            await logger.aerror(
                "neo4j_store_task_result_failed",
                task_id=task_id,
                agent_id=agent_id,
                error=str(exc),
            )
            raise

    def is_connected(self) -> bool:
        """Check if Neo4j is connected.

        Returns:
            True if connected, False otherwise.
        """
        return self._is_connected
