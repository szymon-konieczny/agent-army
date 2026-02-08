"""Recursive Language Model (RLM) engine for handling large contexts.

Based on MIT's RLM paradigm (arXiv 2512.24601), this engine treats long prompts
as external data and uses recursive sub-queries to process them efficiently.

Key concepts:
- Large documents are partitioned intelligently
- Queries are decomposed recursively
- Sub-results are aggregated into final answers
- Works with ANY model client (Claude, OpenAI, HuggingFace, Ollama, Bielik)
"""

import asyncio
import uuid
import re
import time
from typing import Optional, Any, Callable, Awaitable
import structlog
from pydantic import BaseModel, Field

from src.models.schemas import LLMRequest, LLMResponse, ModelProvider


logger = structlog.get_logger(__name__)


class ContextPartition(BaseModel):
    """A partition of a large document context."""

    partition_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = Field(..., description="ID of the parent context")
    content: str = Field(..., description="Text content of this partition")
    start_char: int = Field(..., description="Character position in original text")
    end_char: int = Field(..., description="End character position")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @property
    def char_count(self) -> int:
        """Get character count of this partition."""
        return len(self.content)


class ContextRegistry(BaseModel):
    """Storage for large document contexts."""

    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_text: str = Field(..., description="Full original text")
    partitions: list[ContextPartition] = Field(
        default_factory=list, description="Partitions of the text"
    )
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Context metadata"
    )

    @property
    def char_count(self) -> int:
        """Get total character count."""
        return len(self.original_text)


class RLMQuery(BaseModel):
    """A recursive query over a context."""

    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = Field(..., description="ID of context to query")
    question: str = Field(..., description="User's question")
    depth: int = Field(default=0, description="Recursion depth")
    parent_query_id: Optional[str] = Field(default=None, description="Parent query ID")
    subqueries: list["RLMQuery"] = Field(
        default_factory=list, description="Child queries"
    )
    answer: Optional[str] = Field(default=None, description="Answer for this query")
    tokens_used: int = Field(default=0, description="Tokens used for this query")
    status: str = Field(default="pending", description="Query status")


class RLMConfig(BaseModel):
    """Configuration for RLM engine."""

    max_recursion_depth: int = Field(default=4, description="Maximum recursion depth")
    chunk_size: int = Field(default=4096, description="Context chunk size in characters")
    chunk_overlap: int = Field(
        default=512, description="Overlap between chunks in characters"
    )
    partition_strategy: str = Field(
        default="semantic", description="Partitioning strategy (semantic or fixed)"
    )
    temperature: float = Field(
        default=0.3, description="Temperature for RLM queries (lower=more focused)"
    )
    answer_max_tokens: int = Field(
        default=1024, description="Max tokens for answer generation"
    )
    subquery_max_tokens: int = Field(
        default=512, description="Max tokens for subquery responses"
    )


class RLMEngine:
    """Recursive Language Model engine for large context handling.

    Enables processing of contexts larger than typical model windows by:
    1. Partitioning large texts intelligently
    2. Decomposing queries recursively
    3. Aggregating results from sub-queries
    4. Working with any model client asynchronously
    """

    def __init__(self, config: Optional[RLMConfig] = None) -> None:
        """Initialize RLM engine.

        Args:
            config: RLM configuration options.
        """
        self.config = config or RLMConfig()
        self.contexts: dict[str, ContextRegistry] = {}
        self.queries: dict[str, RLMQuery] = {}

        logger.info(
            "RLMEngine initialized",
            max_recursion_depth=self.config.max_recursion_depth,
            chunk_size=self.config.chunk_size,
            partition_strategy=self.config.partition_strategy,
        )

    async def load_context(
        self, text: str, metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """Load a large text as a searchable context.

        Args:
            text: Large text to load as context.
            metadata: Optional metadata about the context (e.g., source, type).

        Returns:
            Context ID for later reference.
        """
        context = ContextRegistry(original_text=text, metadata=metadata or {})

        # Partition the context
        partitions = await self.partition_context(
            text, strategy=self.config.partition_strategy
        )
        context.partitions = partitions

        # Store context
        self.contexts[context.context_id] = context

        logger.info(
            "Context loaded",
            context_id=context.context_id,
            char_count=context.char_count,
            partition_count=len(partitions),
        )

        return context.context_id

    async def partition_context(
        self, text: str, strategy: str = "fixed"
    ) -> list[ContextPartition]:
        """Partition large text into smaller chunks.

        Args:
            text: Text to partition.
            strategy: Partitioning strategy ("fixed" or "semantic").

        Returns:
            List of context partitions.
        """
        partitions: list[ContextPartition] = []

        if strategy == "fixed":
            # Fixed-size chunking with overlap
            chunk_size = self.config.chunk_size
            overlap = self.config.chunk_overlap

            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i : i + chunk_size]

                if chunk:
                    partition = ContextPartition(
                        context_id="",  # Will be set when storing
                        content=chunk,
                        start_char=i,
                        end_char=min(i + chunk_size, len(text)),
                    )
                    partitions.append(partition)

            logger.debug(
                "Context partitioned (fixed)",
                partition_count=len(partitions),
                avg_size=sum(p.char_count for p in partitions) // len(partitions)
                if partitions
                else 0,
            )

        elif strategy == "semantic":
            # Semantic partitioning (split at sentence boundaries)
            sentences = re.split(r"(?<=[.!?])\s+", text)
            current_chunk = ""
            current_start = 0

            for sentence in sentences:
                if (
                    len(current_chunk) + len(sentence)
                    < self.config.chunk_size
                ):
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        partition = ContextPartition(
                            context_id="",
                            content=current_chunk.strip(),
                            start_char=current_start,
                            end_char=current_start + len(current_chunk),
                        )
                        partitions.append(partition)
                        current_start += len(current_chunk)

                    current_chunk = sentence + " "

            # Don't forget the last chunk
            if current_chunk:
                partition = ContextPartition(
                    context_id="",
                    content=current_chunk.strip(),
                    start_char=current_start,
                    end_char=len(text),
                )
                partitions.append(partition)

            logger.debug(
                "Context partitioned (semantic)",
                partition_count=len(partitions),
            )

        return partitions

    async def recursive_query(
        self,
        question: str,
        context_id: str,
        model_client: Any,
        depth: int = 0,
        parent_query_id: Optional[str] = None,
    ) -> RLMQuery:
        """Recursively query a context.

        The query is decomposed into sub-queries if:
        1. Context is large (multiple partitions)
        2. Recursion depth < max_recursion_depth

        Args:
            question: The user's question.
            context_id: ID of the context to query.
            model_client: Model client to use (must have `complete` method).
            depth: Current recursion depth.
            parent_query_id: ID of parent query (if recursive).

        Returns:
            RLMQuery with results and sub-queries.

        Raises:
            ValueError: If context_id not found.
        """
        if context_id not in self.contexts:
            raise ValueError(f"Context {context_id} not found")

        context = self.contexts[context_id]
        query = RLMQuery(
            context_id=context_id,
            question=question,
            depth=depth,
            parent_query_id=parent_query_id,
        )
        self.queries[query.query_id] = query

        logger.info(
            "Starting recursive query",
            query_id=query.query_id,
            depth=depth,
            partitions=len(context.partitions),
        )

        # Base case: single partition or reached max depth
        if len(context.partitions) == 1 or depth >= self.config.max_recursion_depth:
            answer = await self._answer_from_context(
                question, context.original_text, model_client
            )
            query.answer = answer
            query.status = "completed"
            logger.debug(
                "Query answered (base case)",
                query_id=query.query_id,
                answer_length=len(answer),
            )
            return query

        # Recursive case: spawn sub-queries for each partition
        logger.info(
            "Decomposing query into sub-queries",
            query_id=query.query_id,
            partition_count=len(context.partitions),
        )

        subquery_answers = []
        for partition in context.partitions:
            # Create focused question for this partition
            subquestion = await self._create_subquestion(
                question, partition.content[:500], model_client
            )

            # Recursively query this partition
            subquery = RLMQuery(
                context_id=context_id,
                question=subquestion,
                depth=depth + 1,
                parent_query_id=query.query_id,
            )

            answer = await self._answer_from_context(
                subquestion, partition.content, model_client
            )
            subquery.answer = answer
            subquery.status = "completed"

            query.subqueries.append(subquery)
            subquery_answers.append(answer)

            logger.debug(
                "Sub-query completed",
                query_id=query.query_id,
                subquery_id=subquery.query_id,
                depth=depth + 1,
            )

        # Aggregate sub-query answers into final answer
        aggregated = await self._aggregate_answers(
            question, subquery_answers, model_client
        )
        query.answer = aggregated
        query.status = "completed"

        logger.info(
            "Recursive query completed",
            query_id=query.query_id,
            depth=depth,
            subquery_count=len(query.subqueries),
        )

        return query

    async def search_context(
        self, context_id: str, pattern: str, search_type: str = "regex"
    ) -> list[ContextPartition]:
        """Search within a context using pattern matching.

        Args:
            context_id: ID of context to search.
            pattern: Search pattern (regex or literal string).
            search_type: "regex" or "literal".

        Returns:
            List of matching partitions.

        Raises:
            ValueError: If context_id not found.
        """
        if context_id not in self.contexts:
            raise ValueError(f"Context {context_id} not found")

        context = self.contexts[context_id]
        matches: list[ContextPartition] = []

        try:
            if search_type == "regex":
                regex = re.compile(pattern, re.IGNORECASE)
                for partition in context.partitions:
                    if regex.search(partition.content):
                        matches.append(partition)

            elif search_type == "literal":
                pattern_lower = pattern.lower()
                for partition in context.partitions:
                    if pattern_lower in partition.content.lower():
                        matches.append(partition)

            logger.info(
                "Context search completed",
                context_id=context_id,
                pattern=pattern[:50],
                matches=len(matches),
            )

        except Exception as e:
            logger.error("Context search failed", context_id=context_id, error=str(e))

        return matches

    async def submit_answer(
        self, context_id: str, answer: str, metadata: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Store a final answer associated with a context.

        Args:
            context_id: ID of the context.
            answer: The answer to store.
            metadata: Optional metadata (e.g., answer_type, confidence).

        Returns:
            Dictionary with submission details.

        Raises:
            ValueError: If context_id not found.
        """
        if context_id not in self.contexts:
            raise ValueError(f"Context {context_id} not found")

        context = self.contexts[context_id]
        context.metadata["answer"] = answer
        if metadata:
            context.metadata.update(metadata)

        logger.info(
            "Answer submitted",
            context_id=context_id,
            answer_length=len(answer),
        )

        return {
            "context_id": context_id,
            "answer_length": len(answer),
            "timestamp": time.time(),
        }

    def get_context(self, context_id: str) -> Optional[ContextRegistry]:
        """Retrieve a context by ID.

        Args:
            context_id: ID of the context.

        Returns:
            ContextRegistry or None if not found.
        """
        return self.contexts.get(context_id)

    def get_query(self, query_id: str) -> Optional[RLMQuery]:
        """Retrieve a query by ID.

        Args:
            query_id: ID of the query.

        Returns:
            RLMQuery or None if not found.
        """
        return self.queries.get(query_id)

    # Private methods

    async def _answer_from_context(
        self, question: str, context: str, model_client: Any
    ) -> str:
        """Generate an answer from context using the model.

        Args:
            question: The question to answer.
            context: The context to use.
            model_client: Model client with `complete` method.

        Returns:
            The generated answer.
        """
        prompt = f"""Answer the following question using ONLY the provided context.

Context:
{context[:2000]}

Question: {question}

Answer:"""

        request = LLMRequest(
            prompt=prompt,
            max_tokens=self.config.subquery_max_tokens,
            temperature=self.config.temperature,
        )

        try:
            response = await model_client.complete(request)
            return response.content.strip()
        except Exception as e:
            logger.error("Failed to generate answer", error=str(e))
            return ""

    async def _create_subquestion(
        self, original_question: str, partition_preview: str, model_client: Any
    ) -> str:
        """Create a focused subquestion for a specific partition.

        Args:
            original_question: The original question.
            partition_preview: Preview of the partition content.
            model_client: Model client with `complete` method.

        Returns:
            A focused subquestion.
        """
        prompt = f"""Given this original question and a preview of document content,
create a focused follow-up question that asks the same thing but is optimized for
just this section.

Original question: {original_question}

Document preview:
{partition_preview[:500]}

Focused subquestion:"""

        request = LLMRequest(
            prompt=prompt,
            max_tokens=100,
            temperature=0.2,
        )

        try:
            response = await model_client.complete(request)
            return response.content.strip()
        except Exception as e:
            logger.error("Failed to create subquestion", error=str(e))
            return original_question

    async def _aggregate_answers(
        self, question: str, answers: list[str], model_client: Any
    ) -> str:
        """Aggregate multiple sub-answers into a final answer.

        Args:
            question: The original question.
            answers: List of sub-answers.
            model_client: Model client with `complete` method.

        Returns:
            Aggregated final answer.
        """
        answers_text = "\n\n".join(
            [f"Section {i+1}: {answer}" for i, answer in enumerate(answers)]
        )

        prompt = f"""Combine these answers from different sections of a document into
a single, coherent answer to the original question.

Original question: {question}

Answers from different sections:
{answers_text}

Final synthesized answer:"""

        request = LLMRequest(
            prompt=prompt,
            max_tokens=self.config.answer_max_tokens,
            temperature=self.config.temperature,
        )

        try:
            response = await model_client.complete(request)
            return response.content.strip()
        except Exception as e:
            logger.error("Failed to aggregate answers", error=str(e))
            # Fallback: return first non-empty answer
            return next((a for a in answers if a), "")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about loaded contexts and processed queries.

        Returns:
            Dictionary with statistics.
        """
        total_chars = sum(c.char_count for c in self.contexts.values())
        total_partitions = sum(
            len(c.partitions) for c in self.contexts.values()
        )

        return {
            "contexts_loaded": len(self.contexts),
            "total_characters": total_chars,
            "total_partitions": total_partitions,
            "queries_processed": len(self.queries),
            "completed_queries": sum(
                1 for q in self.queries.values() if q.status == "completed"
            ),
        }
