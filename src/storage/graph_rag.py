"""GraphRAG: Graph-enhanced Retrieval Augmented Generation.

Combines Neo4j knowledge graph traversal with hypergraph analysis
to provide rich, structured context to LLM queries.
"""

import os
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class GraphRAGEngine:
    """Graph-based RAG engine combining Neo4j and hypergraph for LLM augmentation.

    Provides:
    - Question understanding with entity extraction
    - Multi-modal graph retrieval (Neo4j + hypergraph)
    - Context assembly with source tracking
    - Codebase and document indexing
    - Reasoning explanation

    Attributes:
        neo4j_store: Neo4j knowledge graph store.
        hypergraph_store: Hypergraph store for n-ary relations.
        llm_router: LLM routing interface for embeddings and generation.
    """

    def __init__(
        self,
        neo4j_store,
        hypergraph_store,
        llm_router,
    ) -> None:
        """Initialize GraphRAG engine.

        Args:
            neo4j_store: Neo4jStore instance for knowledge graph operations.
            hypergraph_store: HypergraphStore instance for n-ary relations.
            llm_router: LLM router for embeddings and text generation.

        Raises:
            ValueError: If any parameter is None.
        """
        if neo4j_store is None or hypergraph_store is None or llm_router is None:
            raise ValueError("neo4j_store, hypergraph_store, and llm_router are required")

        self.neo4j_store = neo4j_store
        self.hypergraph_store = hypergraph_store
        self.llm_router = llm_router
        self._indexed_documents: dict[str, dict[str, Any]] = {}

    async def query(
        self,
        question: str,
        context_filter: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Query with graph-enhanced RAG retrieval.

        Process:
        1. Extract entities from question
        2. Find relevant nodes in Neo4j
        3. Expand context via graph traversal
        4. Enrich with hypergraph facts
        5. Assemble context and send to LLM
        6. Return answer with source tracking

        Args:
            question: User query/question.
            context_filter: Optional filters for retrieval (e.g., label filters).

        Returns:
            Dictionary with answer, context, and source information.

        Raises:
            RuntimeError: If stores not connected.
        """
        if not self.neo4j_store.is_connected():
            raise RuntimeError("Neo4j store not connected")

        try:
            await logger.ainfo(
                "graph_rag_query_started",
                question=question[:100],
            )

            # Step 1: Extract entities from question
            entities = await self._extract_entities(question)

            await logger.adebug(
                "graph_rag_entities_extracted",
                entity_count=len(entities),
            )

            # Step 2: Find relevant nodes in Neo4j
            relevant_nodes = await self._find_relevant_nodes(
                entities,
                context_filter,
            )

            # Step 3: Expand context via graph traversal
            expanded_context = await self._expand_context(relevant_nodes)

            # Step 4: Enrich with hypergraph facts
            enriched_context = await self._enrich_with_hypergraph_facts(
                expanded_context,
                entities,
            )

            # Step 5: Assemble context
            assembled_context = self._assemble_context_text(enriched_context)

            # Step 6: Generate answer with LLM
            answer = await self._generate_answer(question, assembled_context)

            result = {
                "question": question,
                "answer": answer,
                "context": assembled_context,
                "sources": enriched_context.get("source_nodes", []),
                "entities_found": entities,
                "context_metadata": {
                    "node_count": len(relevant_nodes),
                    "expanded_node_count": len(enriched_context.get("nodes", [])),
                    "fact_count": len(enriched_context.get("facts", [])),
                },
            }

            await logger.ainfo(
                "graph_rag_query_completed",
                question=question[:100],
            )

            return result

        except Exception as exc:
            await logger.aerror(
                "graph_rag_query_failed",
                question=question[:100],
                error=str(exc),
            )
            raise

    async def _extract_entities(self, question: str) -> list[str]:
        """Extract named entities from question using LLM.

        Args:
            question: Question text.

        Returns:
            List of extracted entities.
        """
        try:
            # Use LLM to extract entities
            prompt = f"""Extract key entities from this question. Return as a list.

Question: {question}

Entities (comma-separated):"""

            response = await self.llm_router.generate(
                prompt=prompt,
                max_tokens=100,
            )

            # Parse entity list
            entities = [
                e.strip() for e in response.split(",") if e.strip()
            ]

            await logger.adebug(
                "graph_rag_entities_extracted_from_llm",
                entity_count=len(entities),
            )

            return entities

        except Exception as exc:
            await logger.awarning(
                "graph_rag_entity_extraction_failed",
                error=str(exc),
            )
            return []

    async def _find_relevant_nodes(
        self,
        entities: list[str],
        context_filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Find relevant nodes matching entities.

        Args:
            entities: List of entity names.
            context_filter: Optional filter constraints.

        Returns:
            List of relevant nodes.
        """
        try:
            relevant_nodes = []

            for entity in entities:
                # Search for node by name/properties
                cypher = """
                MATCH (n)
                WHERE n.name CONTAINS $entity OR n.id CONTAINS $entity
                RETURN n
                LIMIT 5
                """

                records = await self.neo4j_store.query(cypher, {"entity": entity})

                for record in records:
                    if "n" in record:
                        relevant_nodes.append(record["n"])

            await logger.adebug(
                "graph_rag_relevant_nodes_found",
                entity_count=len(entities),
                node_count=len(relevant_nodes),
            )

            return relevant_nodes

        except Exception as exc:
            await logger.awarning(
                "graph_rag_find_relevant_nodes_failed",
                error=str(exc),
            )
            return []

    async def _expand_context(
        self,
        seed_nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Expand context via graph traversal from seed nodes.

        Args:
            seed_nodes: Initial set of relevant nodes.

        Returns:
            Expanded context with nodes and relationships.
        """
        try:
            all_nodes = set()
            relationships = []

            for node in seed_nodes:
                node_id = node.get("id")
                if node_id:
                    all_nodes.add(node_id)

                    # Get neighbors (depth 2)
                    try:
                        neighbors = await self.neo4j_store.get_neighbors(
                            node_id,
                            direction="both",
                        )

                        for neighbor in neighbors:
                            neighbor_id = neighbor.get("id")
                            if neighbor_id:
                                all_nodes.add(neighbor_id)

                                relationships.append({
                                    "from": node_id,
                                    "to": neighbor_id,
                                    "data": neighbor,
                                })
                    except Exception as exc:
                        await logger.adebug(
                            "graph_rag_neighbor_retrieval_skipped",
                            node_id=node_id,
                            error=str(exc),
                        )
                        continue

            context = {
                "source_nodes": list(seed_nodes),
                "nodes": list(all_nodes),
                "relationships": relationships,
            }

            await logger.adebug(
                "graph_rag_context_expanded",
                seed_count=len(seed_nodes),
                expanded_node_count=len(all_nodes),
                relationship_count=len(relationships),
            )

            return context

        except Exception as exc:
            await logger.awarning(
                "graph_rag_expand_context_failed",
                error=str(exc),
            )
            return {
                "source_nodes": seed_nodes,
                "nodes": [n.get("id") for n in seed_nodes if "id" in n],
                "relationships": [],
            }

    async def _enrich_with_hypergraph_facts(
        self,
        context: dict[str, Any],
        entities: list[str],
    ) -> dict[str, Any]:
        """Enrich context with hypergraph facts involving entities.

        Args:
            context: Current context data.
            entities: List of entities to search.

        Returns:
            Enriched context with facts added.
        """
        try:
            facts = []

            for entity in entities:
                # Query facts from hypergraph
                entity_facts = await self.hypergraph_store.query_facts(entity)
                facts.extend(entity_facts)

            context["facts"] = facts

            await logger.adebug(
                "graph_rag_context_enriched_with_facts",
                entity_count=len(entities),
                fact_count=len(facts),
            )

            return context

        except Exception as exc:
            await logger.awarning(
                "graph_rag_enrich_with_facts_failed",
                error=str(exc),
            )
            context["facts"] = []
            return context

    def _assemble_context_text(
        self,
        context: dict[str, Any],
    ) -> str:
        """Assemble structured context into readable text.

        Args:
            context: Structured context data.

        Returns:
            Text representation of context.
        """
        try:
            lines = []

            # Source nodes
            source_nodes = context.get("source_nodes", [])
            if source_nodes:
                lines.append("## Key Nodes")
                for node in source_nodes:
                    node_info = self._format_node(node)
                    lines.append(f"- {node_info}")

            # Relationships
            relationships = context.get("relationships", [])
            if relationships:
                lines.append("\n## Relationships")
                for rel in relationships[:10]:  # Limit to first 10
                    from_id = rel.get("from", "unknown")
                    to_id = rel.get("to", "unknown")
                    lines.append(f"- {from_id} -> {to_id}")

            # Facts
            facts = context.get("facts", [])
            if facts:
                lines.append("\n## Related Facts")
                for fact in facts[:5]:  # Limit to first 5
                    meta = fact.get("metadata", {})
                    fact_str = self._format_fact(meta)
                    if fact_str:
                        lines.append(f"- {fact_str}")

            assembled = "\n".join(lines)

            return assembled

        except Exception as exc:
            logger.error(
                "graph_rag_assemble_context_failed",
                error=str(exc),
            )
            return ""

    def _format_node(self, node: dict[str, Any]) -> str:
        """Format a node for display.

        Args:
            node: Node data.

        Returns:
            Formatted node string.
        """
        node_id = node.get("id", "unknown")
        name = node.get("name", node_id)

        details = []
        if "label" in node:
            details.append(f"type:{node['label']}")
        if "description" in node:
            details.append(f"desc:{node['description'][:50]}")

        detail_str = ", ".join(details)
        return f"{name} ({detail_str})" if detail_str else name

    def _format_fact(self, metadata: dict[str, Any]) -> str:
        """Format a fact for display.

        Args:
            metadata: Fact metadata.

        Returns:
            Formatted fact string.
        """
        relation = metadata.get("relation_type", "related")
        return f"{relation}" if relation else ""

    async def _generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:
        """Generate answer using LLM with context.

        Args:
            question: Original question.
            context: Assembled context.

        Returns:
            Generated answer.
        """
        try:
            prompt = f"""Answer the following question using the provided context.

Context:
{context}

Question: {question}

Answer:"""

            answer = await self.llm_router.generate(
                prompt=prompt,
                max_tokens=500,
            )

            return answer

        except Exception as exc:
            await logger.aerror(
                "graph_rag_generate_answer_failed",
                error=str(exc),
            )
            return "Unable to generate answer. Please try again."

    async def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Index a document into the knowledge graph.

        Breaks document into chunks and creates nodes.

        Args:
            doc_id: Unique document ID.
            content: Document content.
            metadata: Optional document metadata.

        Returns:
            True if indexing successful, False otherwise.

        Raises:
            RuntimeError: If Neo4j not connected.
        """
        if not self.neo4j_store.is_connected():
            raise RuntimeError("Neo4j store not connected")

        try:
            # Create document node
            doc_properties = {
                "id": doc_id,
                "content_length": len(content),
                **(metadata or {}),
            }

            await self.neo4j_store.create_node("Document", doc_properties)

            # Break into chunks and create nodes
            chunks = self._chunk_document(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}:chunk_{i}"

                chunk_properties = {
                    "id": chunk_id,
                    "content": chunk,
                    "chunk_index": i,
                    "document_id": doc_id,
                }

                await self.neo4j_store.create_node("Chunk", chunk_properties)

                # Link chunk to document
                await self.neo4j_store.create_relationship(
                    from_id=doc_id,
                    to_id=chunk_id,
                    rel_type="HAS_CHUNK",
                )

            self._indexed_documents[doc_id] = {
                "metadata": metadata,
                "chunk_count": len(chunks),
            }

            await logger.ainfo(
                "graph_rag_document_indexed",
                doc_id=doc_id,
                chunk_count=len(chunks),
            )

            return True

        except Exception as exc:
            await logger.aerror(
                "graph_rag_index_document_failed",
                doc_id=doc_id,
                error=str(exc),
            )
            return False

    def _chunk_document(
        self,
        content: str,
        chunk_size: int = 500,
    ) -> list[str]:
        """Split document into overlapping chunks.

        Args:
            content: Document content.
            chunk_size: Target chunk size in characters.

        Returns:
            List of text chunks.
        """
        try:
            chunks = []
            paragraphs = content.split("\n\n")

            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks

        except Exception as exc:
            logger.error(
                "graph_rag_chunk_document_failed",
                error=str(exc),
            )
            return [content]

    async def index_codebase(
        self,
        root_path: str,
    ) -> dict[str, int]:
        """Index entire codebase into knowledge graph.

        Walks directory tree, indexes files, functions, and dependencies.

        Args:
            root_path: Root directory path for codebase.

        Returns:
            Dictionary with index statistics.

        Raises:
            RuntimeError: If Neo4j not connected.
            ValueError: If root_path doesn't exist.
        """
        if not self.neo4j_store.is_connected():
            raise RuntimeError("Neo4j store not connected")

        root = Path(root_path)
        if not root.exists():
            raise ValueError(f"Path does not exist: {root_path}")

        try:
            file_count = 0
            function_count = 0
            dependency_count = 0

            # Walk directory tree
            for py_file in root.rglob("*.py"):
                file_count += 1
                filepath = str(py_file)

                # Get file metadata
                file_stat = py_file.stat()
                metadata = {
                    "size": file_stat.st_size,
                    "modified": file_stat.st_mtime,
                }

                # Index file
                await self.neo4j_store.index_file(filepath, metadata)

                # Extract functions from file
                try:
                    functions = self._extract_functions(py_file)
                    for func_name, func_sig, func_doc in functions:
                        func_id = await self.neo4j_store.index_function(
                            name=func_name,
                            filepath=filepath,
                            signature=func_sig,
                            docstring=func_doc,
                        )
                        function_count += 1
                except Exception as exc:
                    await logger.adebug(
                        "graph_rag_function_extraction_failed",
                        filepath=filepath,
                        error=str(exc),
                    )

            await logger.ainfo(
                "graph_rag_codebase_indexed",
                file_count=file_count,
                function_count=function_count,
            )

            return {
                "files_indexed": file_count,
                "functions_indexed": function_count,
                "dependencies_indexed": dependency_count,
            }

        except Exception as exc:
            await logger.aerror(
                "graph_rag_index_codebase_failed",
                root_path=root_path,
                error=str(exc),
            )
            raise

    def _extract_functions(self, py_file: Path) -> list[tuple[str, str, str]]:
        """Extract function definitions from Python file.

        Args:
            py_file: Path to Python file.

        Returns:
            List of (name, signature, docstring) tuples.
        """
        try:
            functions = []

            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple regex-based extraction (could use ast module for better parsing)
            import re

            # Find function definitions
            func_pattern = r"async def (\w+)\s*\((.*?)\).*?:.*?\n(.*?)(?=\n\s{0,4}(?:def|class|async|$))"
            matches = re.finditer(func_pattern, content, re.DOTALL)

            for match in matches:
                name = match.group(1)
                params = match.group(2)
                body = match.group(3).strip()

                # Extract docstring
                docstring = ""
                doc_match = re.search(r'"""(.*?)"""', body, re.DOTALL)
                if doc_match:
                    docstring = doc_match.group(1).strip()

                signature = f"async def {name}({params})"
                functions.append((name, signature, docstring))

            return functions

        except Exception as exc:
            logger.error(
                "graph_rag_extract_functions_failed",
                filepath=str(py_file),
                error=str(exc),
            )
            return []

    async def get_relevant_context(
        self,
        query: str,
        max_nodes: int = 10,
    ) -> dict[str, Any]:
        """Retrieve relevant context without generating answer.

        Args:
            query: Query string.
            max_nodes: Maximum nodes to return.

        Returns:
            Dictionary with relevant context.

        Raises:
            RuntimeError: If Neo4j not connected.
        """
        if not self.neo4j_store.is_connected():
            raise RuntimeError("Neo4j store not connected")

        try:
            entities = await self._extract_entities(query)
            relevant_nodes = await self._find_relevant_nodes(entities)

            # Limit to max_nodes
            relevant_nodes = relevant_nodes[:max_nodes]

            context = await self._expand_context(relevant_nodes)
            enriched = await self._enrich_with_hypergraph_facts(context, entities)

            return enriched

        except Exception as exc:
            await logger.aerror(
                "graph_rag_get_relevant_context_failed",
                query=query[:100],
                error=str(exc),
            )
            raise

    async def explain_reasoning(
        self,
        query: str,
    ) -> str:
        """Explain the reasoning path that led to answer.

        Args:
            query: Original query.

        Returns:
            Explanation text describing retrieval reasoning.

        Raises:
            RuntimeError: If Neo4j not connected.
        """
        if not self.neo4j_store.is_connected():
            raise RuntimeError("Neo4j store not connected")

        try:
            entities = await self._extract_entities(query)

            explanation_lines = [
                f"Query: {query}",
                f"\nExtracted Entities: {', '.join(entities)}",
            ]

            # Find nodes
            relevant_nodes = await self._find_relevant_nodes(entities)
            explanation_lines.append(
                f"\nFound {len(relevant_nodes)} relevant nodes from knowledge graph"
            )

            # Explain traversal
            explanation_lines.append("\nGraph Traversal:")
            for node in relevant_nodes[:3]:
                node_id = node.get("id", "unknown")
                explanation_lines.append(f"  - Retrieved node: {node_id}")

                try:
                    neighbors = await self.neo4j_store.get_neighbors(node_id)
                    explanation_lines.append(
                        f"    - Found {len(neighbors)} connected nodes"
                    )
                except Exception:
                    pass

            explanation = "\n".join(explanation_lines)

            await logger.adebug(
                "graph_rag_reasoning_explained",
                query=query[:100],
            )

            return explanation

        except Exception as exc:
            await logger.aerror(
                "graph_rag_explain_reasoning_failed",
                query=query[:100],
                error=str(exc),
            )
            return "Unable to explain reasoning."
