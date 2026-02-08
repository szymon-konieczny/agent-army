"""Hypergraph store for modeling complex multi-entity relationships.

Built on HyperNetX. Used alongside Neo4j for higher-order analysis of
agent collaboration, knowledge representation, and fact relations.
"""

import json
from typing import Any, Optional
from collections import defaultdict

import structlog

logger = structlog.get_logger(__name__)


class HypergraphStore:
    """Hypergraph store for n-ary relationships and complex interactions.

    Provides:
    - N-ary relationship representation (hyperedges)
    - Agent collaboration tracking and team effectiveness
    - Knowledge representation as multi-entity facts
    - Analytics on hypergraph structure
    - Persistence and Neo4j synchronization

    Attributes:
        hypergraph: Internal hypergraph data structure (dict-based)
        nodes: Set of all nodes in hypergraph
        hyperedges: Dictionary of hyperedges with metadata
    """

    def __init__(self) -> None:
        """Initialize empty hypergraph store."""
        self.hypergraph: dict[str, Any] = {
            "nodes": set(),
            "hyperedges": {},  # edge_id -> {nodes, metadata}
        }
        self.nodes: set[str] = set()
        self.hyperedges: dict[str, dict[str, Any]] = {}

    async def add_hyperedge(
        self,
        edge_id: str,
        nodes: list[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Add a hyperedge (n-ary relationship) to the hypergraph.

        Args:
            edge_id: Unique identifier for the hyperedge.
            nodes: List of node IDs participating in this hyperedge.
            metadata: Optional metadata about the hyperedge.

        Returns:
            True if hyperedge added successfully, False otherwise.

        Raises:
            ValueError: If edge_id or nodes list is empty.
        """
        if not edge_id or not nodes:
            raise ValueError("edge_id and nodes list cannot be empty")

        if len(nodes) < 2:
            raise ValueError("Hyperedge must contain at least 2 nodes")

        try:
            # Add nodes to hypergraph
            for node in nodes:
                self.nodes.add(node)
                self.hypergraph["nodes"].add(node)

            # Create hyperedge
            self.hyperedges[edge_id] = {
                "nodes": set(nodes),
                "metadata": metadata or {},
                "node_count": len(nodes),
            }

            self.hypergraph["hyperedges"][edge_id] = self.hyperedges[edge_id]

            await logger.adebug(
                "hypergraph_hyperedge_added",
                edge_id=edge_id,
                node_count=len(nodes),
                metadata_keys=list((metadata or {}).keys()),
            )

            return True

        except Exception as exc:
            await logger.aerror(
                "hypergraph_add_hyperedge_failed",
                edge_id=edge_id,
                error=str(exc),
            )
            return False

    async def remove_hyperedge(self, edge_id: str) -> bool:
        """Remove a hyperedge from the hypergraph.

        Args:
            edge_id: ID of hyperedge to remove.

        Returns:
            True if removed, False if not found.
        """
        try:
            if edge_id in self.hyperedges:
                del self.hyperedges[edge_id]
                del self.hypergraph["hyperedges"][edge_id]

                await logger.adebug(
                    "hypergraph_hyperedge_removed",
                    edge_id=edge_id,
                )

                return True

            return False

        except Exception as exc:
            await logger.aerror(
                "hypergraph_remove_hyperedge_failed",
                edge_id=edge_id,
                error=str(exc),
            )
            return False

    async def get_hyperedges_for_node(
        self,
        node_id: str,
    ) -> list[dict[str, Any]]:
        """Get all hyperedges containing a specific node.

        Args:
            node_id: ID of node to search for.

        Returns:
            List of hyperedges containing this node.
        """
        try:
            result = []

            for edge_id, edge_data in self.hyperedges.items():
                if node_id in edge_data["nodes"]:
                    result.append({
                        "edge_id": edge_id,
                        **edge_data,
                    })

            await logger.adebug(
                "hypergraph_node_hyperedges_retrieved",
                node_id=node_id,
                edge_count=len(result),
            )

            return result

        except Exception as exc:
            await logger.aerror(
                "hypergraph_get_hyperedges_for_node_failed",
                node_id=node_id,
                error=str(exc),
            )
            return []

    async def get_shared_hyperedges(
        self,
        nodes: list[str],
    ) -> list[dict[str, Any]]:
        """Find hyperedges that connect ALL given nodes.

        Args:
            nodes: List of node IDs to find common hyperedges for.

        Returns:
            List of hyperedges containing all given nodes.
        """
        if not nodes:
            return []

        try:
            nodes_set = set(nodes)
            result = []

            for edge_id, edge_data in self.hyperedges.items():
                # Check if all nodes are in this hyperedge
                if nodes_set.issubset(edge_data["nodes"]):
                    result.append({
                        "edge_id": edge_id,
                        **edge_data,
                    })

            await logger.adebug(
                "hypergraph_shared_hyperedges_retrieved",
                node_count=len(nodes),
                shared_edge_count=len(result),
            )

            return result

        except Exception as exc:
            await logger.aerror(
                "hypergraph_get_shared_hyperedges_failed",
                node_count=len(nodes),
                error=str(exc),
            )
            return []

    async def record_collaboration(
        self,
        task_id: str,
        agent_ids: list[str],
        outcome: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Record agent collaboration as a hyperedge.

        Args:
            task_id: ID of task/objective.
            agent_ids: List of participating agent IDs.
            outcome: Result outcome (success, failure, partial).
            metadata: Additional collaboration metadata.

        Returns:
            True if collaboration recorded, False otherwise.
        """
        if not agent_ids or len(agent_ids) < 2:
            await logger.awarning(
                "hypergraph_collaboration_requires_multiple_agents",
                agent_count=len(agent_ids),
            )
            return False

        try:
            edge_id = f"collab_{task_id}_{hash(str(sorted(agent_ids)))}"

            meta = {
                "task_id": task_id,
                "outcome": outcome,
                "agent_count": len(agent_ids),
                **(metadata or {}),
            }

            return await self.add_hyperedge(edge_id, agent_ids, meta)

        except Exception as exc:
            await logger.aerror(
                "hypergraph_record_collaboration_failed",
                task_id=task_id,
                agent_count=len(agent_ids),
                error=str(exc),
            )
            return False

    async def get_effective_teams(
        self,
        min_success_rate: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Find agent combinations that work well together.

        Args:
            min_success_rate: Minimum success rate threshold (0.0-1.0).

        Returns:
            List of effective team compositions.
        """
        try:
            if not self.hyperedges:
                return []

            # Group collaborations by agent set
            team_stats: dict[str, dict[str, int]] = defaultdict(
                lambda: {"success": 0, "total": 0}
            )

            for edge_id, edge_data in self.hyperedges.items():
                meta = edge_data.get("metadata", {})

                if "outcome" in meta and "agent_count" in meta:
                    team_key = f"team_{meta['agent_count']}"
                    team_stats[team_key]["total"] += 1

                    if meta["outcome"] == "success":
                        team_stats[team_key]["success"] += 1

            # Filter by success rate
            effective_teams = []

            for team_key, stats in team_stats.items():
                if stats["total"] > 0:
                    success_rate = stats["success"] / stats["total"]

                    if success_rate >= min_success_rate:
                        effective_teams.append({
                            "team": team_key,
                            "success_rate": success_rate,
                            "total_collaborations": stats["total"],
                            "successful_outcomes": stats["success"],
                        })

            # Sort by success rate descending
            effective_teams.sort(key=lambda x: x["success_rate"], reverse=True)

            await logger.adebug(
                "hypergraph_effective_teams_analyzed",
                team_count=len(effective_teams),
                min_success_rate=min_success_rate,
            )

            return effective_teams

        except Exception as exc:
            await logger.aerror(
                "hypergraph_get_effective_teams_failed",
                error=str(exc),
            )
            return []

    async def get_agent_collaboration_score(
        self,
        agent_ids: list[str],
    ) -> float:
        """Calculate collaboration effectiveness score for a team.

        Args:
            agent_ids: List of agent IDs to score.

        Returns:
            Collaboration score between 0.0 and 1.0.
        """
        if not agent_ids or len(agent_ids) < 2:
            return 0.0

        try:
            agent_set = set(agent_ids)
            total_collaborations = 0
            successful_collaborations = 0

            # Find hyperedges containing all agents
            for edge_id, edge_data in self.hyperedges.items():
                if agent_set.issubset(edge_data["nodes"]):
                    meta = edge_data.get("metadata", {})
                    total_collaborations += 1

                    if meta.get("outcome") == "success":
                        successful_collaborations += 1

            if total_collaborations == 0:
                return 0.0

            score = successful_collaborations / total_collaborations

            await logger.adebug(
                "hypergraph_collaboration_score_calculated",
                agent_count=len(agent_ids),
                score=score,
                total_collaborations=total_collaborations,
            )

            return score

        except Exception as exc:
            await logger.aerror(
                "hypergraph_get_collaboration_score_failed",
                agent_count=len(agent_ids),
                error=str(exc),
            )
            return 0.0

    async def add_fact(
        self,
        entities: list[str],
        relation_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Add a multi-entity fact as a hyperedge.

        Args:
            entities: List of entities involved in the fact.
            relation_type: Type of relationship.
            metadata: Additional fact metadata.

        Returns:
            True if fact added successfully, False otherwise.
        """
        try:
            fact_id = f"fact_{relation_type}_{hash(str(sorted(entities)))}"

            meta = {
                "relation_type": relation_type,
                **(metadata or {}),
            }

            return await self.add_hyperedge(fact_id, entities, meta)

        except Exception as exc:
            await logger.aerror(
                "hypergraph_add_fact_failed",
                relation_type=relation_type,
                entity_count=len(entities),
                error=str(exc),
            )
            return False

    async def query_facts(
        self,
        entity: str,
        relation_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Find facts involving a specific entity.

        Args:
            entity: Entity to search for.
            relation_type: Filter by relation type (optional).

        Returns:
            List of facts containing the entity.
        """
        try:
            result = []

            for edge_id, edge_data in self.hyperedges.items():
                if entity in edge_data["nodes"]:
                    meta = edge_data.get("metadata", {})

                    # Filter by relation type if specified
                    if relation_type and meta.get("relation_type") != relation_type:
                        continue

                    result.append({
                        "edge_id": edge_id,
                        **edge_data,
                    })

            await logger.adebug(
                "hypergraph_facts_queried",
                entity=entity,
                relation_type=relation_type,
                fact_count=len(result),
            )

            return result

        except Exception as exc:
            await logger.aerror(
                "hypergraph_query_facts_failed",
                entity=entity,
                relation_type=relation_type,
                error=str(exc),
            )
            return []

    async def find_bridging_entities(
        self,
        entity_a: str,
        entity_b: str,
    ) -> list[str]:
        """Find entities that connect two given entities through hyperedges.

        Args:
            entity_a: First entity ID.
            entity_b: Second entity ID.

        Returns:
            List of bridging entity IDs.
        """
        try:
            # Find hyperedges containing A
            edges_with_a = set()
            for edge_id, edge_data in self.hyperedges.items():
                if entity_a in edge_data["nodes"]:
                    edges_with_a.add(edge_id)

            # Find hyperedges containing B
            edges_with_b = set()
            for edge_id, edge_data in self.hyperedges.items():
                if entity_b in edge_data["nodes"]:
                    edges_with_b.add(edge_id)

            # Find common edges
            common_edges = edges_with_a.intersection(edges_with_b)

            # Collect bridging entities from common edges
            bridging = set()
            for edge_id in common_edges:
                edge_data = self.hyperedges[edge_id]
                nodes = edge_data["nodes"] - {entity_a, entity_b}
                bridging.update(nodes)

            result = list(bridging)

            await logger.adebug(
                "hypergraph_bridging_entities_found",
                entity_a=entity_a,
                entity_b=entity_b,
                bridging_count=len(result),
            )

            return result

        except Exception as exc:
            await logger.aerror(
                "hypergraph_find_bridging_entities_failed",
                entity_a=entity_a,
                entity_b=entity_b,
                error=str(exc),
            )
            return []

    async def get_node_degree(self, node_id: str) -> int:
        """Get the degree (number of hyperedges) for a node.

        Args:
            node_id: ID of node.

        Returns:
            Number of hyperedges containing this node.
        """
        try:
            degree = 0

            for edge_data in self.hyperedges.values():
                if node_id in edge_data["nodes"]:
                    degree += 1

            await logger.adebug(
                "hypergraph_node_degree_retrieved",
                node_id=node_id,
                degree=degree,
            )

            return degree

        except Exception as exc:
            await logger.aerror(
                "hypergraph_get_node_degree_failed",
                node_id=node_id,
                error=str(exc),
            )
            return 0

    async def get_most_connected_nodes(self, n: int = 10) -> list[dict[str, Any]]:
        """Get the top n most connected nodes in the hypergraph.

        Args:
            n: Number of top nodes to return.

        Returns:
            List of nodes with degree information, sorted by degree descending.
        """
        try:
            node_degrees: dict[str, int] = {}

            for node_id in self.nodes:
                node_degrees[node_id] = await self.get_node_degree(node_id)

            # Sort by degree descending and take top n
            sorted_nodes = sorted(
                node_degrees.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:n]

            result = [
                {
                    "node_id": node_id,
                    "degree": degree,
                }
                for node_id, degree in sorted_nodes
            ]

            await logger.adebug(
                "hypergraph_most_connected_nodes_retrieved",
                requested_count=n,
                returned_count=len(result),
            )

            return result

        except Exception as exc:
            await logger.aerror(
                "hypergraph_get_most_connected_nodes_failed",
                n=n,
                error=str(exc),
            )
            return []

    async def get_hyperedge_size_distribution(self) -> dict[int, int]:
        """Get distribution of hyperedge sizes.

        Returns:
            Dictionary mapping edge size to count of edges with that size.
        """
        try:
            distribution: dict[int, int] = {}

            for edge_data in self.hyperedges.values():
                size = edge_data["node_count"]
                distribution[size] = distribution.get(size, 0) + 1

            await logger.adebug(
                "hypergraph_edge_size_distribution_retrieved",
                size_count=len(distribution),
            )

            return distribution

        except Exception as exc:
            await logger.aerror(
                "hypergraph_get_edge_size_distribution_failed",
                error=str(exc),
            )
            return {}

    async def export_to_json(self) -> str:
        """Export hypergraph to JSON format.

        Returns:
            JSON string representation of hypergraph.
        """
        try:
            export_data = {
                "nodes": list(self.nodes),
                "hyperedges": {
                    edge_id: {
                        "nodes": list(edge_data["nodes"]),
                        "metadata": edge_data["metadata"],
                        "node_count": edge_data["node_count"],
                    }
                    for edge_id, edge_data in self.hyperedges.items()
                },
            }

            json_str = json.dumps(export_data, indent=2)

            await logger.adebug(
                "hypergraph_exported_to_json",
                node_count=len(self.nodes),
                edge_count=len(self.hyperedges),
            )

            return json_str

        except Exception as exc:
            await logger.aerror(
                "hypergraph_export_to_json_failed",
                error=str(exc),
            )
            raise

    async def import_from_json(self, json_str: str) -> bool:
        """Import hypergraph from JSON format.

        Args:
            json_str: JSON string representation of hypergraph.

        Returns:
            True if import successful, False otherwise.
        """
        try:
            data = json.loads(json_str)

            # Import nodes
            for node_id in data.get("nodes", []):
                self.nodes.add(node_id)
                self.hypergraph["nodes"].add(node_id)

            # Import hyperedges
            for edge_id, edge_info in data.get("hyperedges", {}).items():
                nodes = edge_info.get("nodes", [])
                metadata = edge_info.get("metadata", {})

                await self.add_hyperedge(edge_id, nodes, metadata)

            await logger.adebug(
                "hypergraph_imported_from_json",
                node_count=len(self.nodes),
                edge_count=len(self.hyperedges),
            )

            return True

        except Exception as exc:
            await logger.aerror(
                "hypergraph_import_from_json_failed",
                error=str(exc),
            )
            return False

    async def sync_from_neo4j(
        self,
        neo4j_store,
        query: str,
    ) -> bool:
        """Populate hypergraph from Neo4j query results.

        Args:
            neo4j_store: Neo4jStore instance.
            query: Cypher query to execute.

        Returns:
            True if sync successful, False otherwise.
        """
        try:
            # Execute query on Neo4j
            records = await neo4j_store.query(query)

            # Process records into hyperedges
            for record in records:
                # Expect records with 'nodes' and optional 'relation_type'
                if "nodes" in record:
                    nodes = record.get("nodes", [])
                    metadata = {
                        k: v for k, v in record.items() if k != "nodes"
                    }

                    edge_id = f"neo4j_{hash(str(sorted(nodes)))}"
                    await self.add_hyperedge(edge_id, nodes, metadata)

            await logger.adebug(
                "hypergraph_synced_from_neo4j",
                record_count=len(records),
                edge_count=len(self.hyperedges),
            )

            return True

        except Exception as exc:
            await logger.aerror(
                "hypergraph_sync_from_neo4j_failed",
                error=str(exc),
            )
            return False

    async def export_insights_to_neo4j(
        self,
        neo4j_store,
    ) -> bool:
        """Export discovered patterns and insights back to Neo4j.

        Args:
            neo4j_store: Neo4jStore instance.

        Returns:
            True if export successful, False otherwise.
        """
        try:
            # Get effective teams
            effective_teams = await self.get_effective_teams()

            # Create Pattern nodes for each effective team
            for team in effective_teams:
                team_id = f"pattern_{team['team']}"
                properties = {
                    "id": team_id,
                    "team": team["team"],
                    "success_rate": team["success_rate"],
                    "total_collaborations": team["total_collaborations"],
                    "successful_outcomes": team["successful_outcomes"],
                }

                await neo4j_store.create_node("Pattern", properties)

            # Export most connected nodes
            top_nodes = await self.get_most_connected_nodes(10)

            for node_info in top_nodes:
                node_id = node_info["node_id"]
                cypher = """
                MATCH (n {id: $node_id})
                SET n.hypergraph_degree = $degree
                RETURN n
                """

                params = {
                    "node_id": node_id,
                    "degree": node_info["degree"],
                }

                await neo4j_store.query(cypher, params)

            await logger.adebug(
                "hypergraph_insights_exported_to_neo4j",
                pattern_count=len(effective_teams),
                node_count=len(top_nodes),
            )

            return True

        except Exception as exc:
            await logger.aerror(
                "hypergraph_export_insights_to_neo4j_failed",
                error=str(exc),
            )
            return False
