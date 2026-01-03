"""
Crypto RAG (Retrieval-Augmented Generation) module for the Crypto Narrative Pulse Tracker.

Extends the Pathway RAG template's question answerer with:
- Context-enriched prompts using live metrics (pulse_score, trending_phrases, etc.)
- Retrieval of top 15 relevant messages
- Ollama integration for local LLM inference

Requirements: 8.3, 8.4
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from rag.live_metrics import LiveMetrics, get_live_metrics

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

CRYPTO_RAG_SYSTEM_PROMPT = """You are a crypto momentum analyst assistant. Your role is to analyze social media sentiment and provide actionable insights for crypto traders.

You have access to:
1. Real-time social media messages about cryptocurrencies
2. Live pulse score (1-10 momentum indicator)
3. Trending phrases from recent discussions
4. Influencer consensus (bullish/bearish/neutral)
5. Price-sentiment divergence status

Always provide:
- Direct, actionable answers
- Risk assessment when relevant
- Data-backed insights from the retrieved messages"""

CRYPTO_RAG_PROMPT_TEMPLATE = """Current Market Pulse:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Pulse Score: {pulse_score}/10
🔥 Trending Phrases: {trending_phrases}
👥 Influencer Consensus: {influencer_consensus}
📈 Price-Sentiment Status: {divergence_status}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recent Social Messages:
{context}

User Question: {query}

Please provide:
1. Direct answer based on the data above
2. Key phrases/narratives driving current sentiment
3. Risk assessment (early opportunity vs late to the narrative)"""

CRYPTO_RAG_SUMMARY_TEMPLATE = """Based on the retrieved messages and current market pulse:

Pulse Score: {pulse_score}/10 - {pulse_interpretation}
Trending: {trending_phrases}
Influencer View: {influencer_consensus}
Divergence: {divergence_status}

{summary}"""


# =============================================================================
# RESPONSE DATA CLASSES
# =============================================================================


@dataclass
class RAGResponse:
    """
    Response from the Crypto RAG system.

    Attributes:
        answer: The generated answer text
        pulse_score: Current pulse score at time of query
        trending_phrases: Trending phrases at time of query
        sources: List of source message metadata
        relevance_scores: Relevance scores for retrieved messages
        influencer_consensus: Influencer consensus at time of query
        divergence_status: Divergence status at time of query
    """

    answer: str
    pulse_score: float
    trending_phrases: list[str]
    sources: list[dict[str, Any]]
    relevance_scores: list[float]
    influencer_consensus: str = "neutral"
    divergence_status: str = "aligned"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "pulse_score": self.pulse_score,
            "trending_phrases": self.trending_phrases,
            "sources": self.sources,
            "relevance_scores": self.relevance_scores,
            "influencer_consensus": self.influencer_consensus,
            "divergence_status": self.divergence_status,
        }


# =============================================================================
# CRYPTO RAG CLASS
# =============================================================================


class CryptoRAG:
    """
    Context-enriched RAG system for crypto sentiment analysis.

    Extends the base Pathway RAG with live metrics context enrichment
    and crypto-specific prompt templates.

    Requirements: 8.3, 8.4

    Example:
        >>> from rag.crypto_rag import CryptoRAG
        >>> rag = CryptoRAG(llm_client=ollama_client, retriever=doc_store)
        >>> response = rag.answer("What's the sentiment on $MEME?")
        >>> print(response.answer)
    """

    # Number of messages to retrieve (Requirement 8.3)
    DEFAULT_TOP_K = 15

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        retriever: Optional[Any] = None,
        top_k: int = DEFAULT_TOP_K,
        live_metrics: Optional[LiveMetrics] = None,
    ):
        """
        Initialize the Crypto RAG system.

        Args:
            llm_client: LLM client for generating responses (e.g., Ollama)
            retriever: Document retriever/vector store for message retrieval
            top_k: Number of messages to retrieve (default: 15)
            live_metrics: LiveMetrics instance (uses singleton if not provided)
        """
        self.llm_client = llm_client
        self.retriever = retriever
        self.top_k = top_k
        self._live_metrics = live_metrics

    @property
    def live_metrics(self) -> LiveMetrics:
        """Get the live metrics instance."""
        if self._live_metrics is None:
            self._live_metrics = get_live_metrics()
        return self._live_metrics

    def _get_pulse_interpretation(self, score: float) -> str:
        """Get human-readable interpretation of pulse score."""
        if score >= 8:
            return "🚀 Very Strong Bullish Momentum"
        elif score >= 7:
            return "📈 Strong Buy Signal"
        elif score >= 5:
            return "➡️ Neutral/Moderate Activity"
        elif score >= 3:
            return "📉 Cooling Off"
        else:
            return "❄️ Low Activity/Bearish"

    def _format_context(self, messages: list[dict[str, Any]]) -> str:
        """Format retrieved messages for prompt context."""
        if not messages:
            return "No recent messages found."

        formatted = []
        for i, msg in enumerate(messages[: self.top_k], 1):
            text = msg.get("text", "")
            author = msg.get("author_id", "unknown")
            sentiment = msg.get("sentiment", 0.0)
            platform = msg.get("source_platform", "unknown")

            sentiment_emoji = (
                "🟢" if sentiment > 0.3 else "🔴" if sentiment < -0.3 else "⚪"
            )

            formatted.append(
                f'{i}. [{platform}] @{author}: "{text[:200]}..." {sentiment_emoji}'
                if len(text) > 200
                else f'{i}. [{platform}] @{author}: "{text}" {sentiment_emoji}'
            )

        return "\n".join(formatted)

    def build_prompt(
        self,
        query: str,
        retrieved_messages: list[dict[str, Any]],
    ) -> str:
        """
        Build context-enriched prompt for LLM.

        Combines live metrics with retrieved messages to create
        a comprehensive prompt for the LLM.

        Args:
            query: User's question
            retrieved_messages: List of retrieved message dictionaries

        Returns:
            Formatted prompt string

        Requirements: 8.4
        """
        # Get current metrics for context enrichment
        rag_context = self.live_metrics.get_rag_context()

        # Format the retrieved messages
        context = self._format_context(retrieved_messages)

        # Build the prompt using template
        prompt = CRYPTO_RAG_PROMPT_TEMPLATE.format(
            pulse_score=rag_context["pulse_score"],
            trending_phrases=rag_context["trending_phrases"],
            influencer_consensus=rag_context["influencer_consensus"],
            divergence_status=rag_context["divergence_status"],
            context=context,
            query=query,
        )

        return prompt

    def retrieve_messages(self, query: str) -> tuple[list[dict[str, Any]], list[float]]:
        """
        Retrieve relevant messages from the document store.

        Args:
            query: Search query

        Returns:
            Tuple of (messages, relevance_scores)

        Requirements: 8.3
        """
        if self.retriever is None:
            logger.warning("No retriever configured, returning empty results")
            return [], []

        try:
            # Use the retriever to get relevant messages
            # The exact API depends on the Pathway VectorStoreServer implementation
            results = self.retriever.query(query, k=self.top_k)

            messages = []
            scores = []

            for result in results:
                if hasattr(result, "text"):
                    msg = {
                        "text": result.text,
                        "metadata": getattr(result, "metadata", {}),
                    }
                    # Flatten metadata into message dict
                    if isinstance(msg["metadata"], dict):
                        msg.update(msg["metadata"])
                    messages.append(msg)
                    scores.append(getattr(result, "score", 0.0))
                elif isinstance(result, dict):
                    messages.append(result)
                    scores.append(result.get("score", 0.0))

            logger.info(
                f"Retrieved {len(messages)} messages for query: {query[:50]}..."
            )
            return messages, scores

        except Exception as e:
            logger.error(f"Error retrieving messages: {e}")
            return [], []

    def generate_response(self, prompt: str) -> str:
        """
        Generate LLM response using Ollama.

        Args:
            prompt: The formatted prompt

        Returns:
            Generated response text
        """
        if self.llm_client is None:
            logger.warning("No LLM client configured, returning placeholder")
            return self._generate_fallback_response()

        try:
            # Call the LLM client
            # The exact API depends on the LLM client implementation
            response = self.llm_client.generate(
                prompt=prompt,
                system=CRYPTO_RAG_SYSTEM_PROMPT,
            )

            if hasattr(response, "text"):
                return response.text
            elif isinstance(response, dict):
                return response.get("text", response.get("content", str(response)))
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._generate_fallback_response()

    def _generate_fallback_response(self) -> str:
        """Generate a fallback response when LLM is unavailable."""
        metrics = self.live_metrics.get_rag_context()
        score = float(metrics["pulse_score"])

        return f"""Based on current metrics:

📊 **Pulse Score: {metrics["pulse_score"]}/10** - {self._get_pulse_interpretation(score)}

🔥 **Trending Phrases:** {metrics["trending_phrases"]}

👥 **Influencer Consensus:** {metrics["influencer_consensus"]}

📈 **Price-Sentiment Status:** {metrics["divergence_status"]}

*Note: LLM analysis unavailable. Showing raw metrics only.*"""

    def answer(self, query: str) -> RAGResponse:
        """
        Answer a user query with context-enriched RAG.

        This is the main entry point for the Crypto RAG system.
        It retrieves relevant messages, enriches the prompt with
        live metrics, and generates a response.

        Args:
            query: User's question

        Returns:
            RAGResponse with answer and metadata

        Requirements: 8.3, 8.4

        Example:
            >>> response = rag.answer("What's the current sentiment on $MEME?")
            >>> print(response.answer)
            >>> print(f"Pulse Score: {response.pulse_score}")
        """
        # Step 1: Retrieve relevant messages (Requirement 8.3)
        messages, scores = self.retrieve_messages(query)

        # Step 2: Build context-enriched prompt (Requirement 8.4)
        prompt = self.build_prompt(query, messages)

        # Step 3: Generate response
        answer = self.generate_response(prompt)

        # Step 4: Get current metrics snapshot
        snapshot = self.live_metrics.get_snapshot()

        # Step 5: Build response object
        return RAGResponse(
            answer=answer,
            pulse_score=snapshot["pulse_score"],
            trending_phrases=snapshot["trending_phrases"],
            sources=[
                {"text": m.get("text", ""), **m.get("metadata", {})} for m in messages
            ],
            relevance_scores=scores,
            influencer_consensus=snapshot["influencer_consensus"],
            divergence_status=snapshot["divergence_status"],
        )

    def answer_with_metrics(
        self,
        query: str,
        pulse_score: Optional[float] = None,
        trending_phrases: Optional[list[str]] = None,
        influencer_consensus: Optional[str] = None,
        divergence_status: Optional[str] = None,
    ) -> RAGResponse:
        """
        Answer a query with explicitly provided metrics.

        Useful for testing or when metrics should be overridden.

        Args:
            query: User's question
            pulse_score: Override pulse score
            trending_phrases: Override trending phrases
            influencer_consensus: Override influencer consensus
            divergence_status: Override divergence status

        Returns:
            RAGResponse with answer and metadata
        """
        # Temporarily update metrics if provided
        metrics = self.live_metrics
        original_snapshot = metrics.get_snapshot()

        try:
            if pulse_score is not None:
                metrics.update(pulse_score=pulse_score)
            if trending_phrases is not None:
                metrics.update(trending_phrases=trending_phrases)
            if influencer_consensus is not None:
                metrics.update(influencer_consensus=influencer_consensus)
            if divergence_status is not None:
                metrics.update(divergence_status=divergence_status)

            return self.answer(query)

        finally:
            # Restore original metrics
            metrics.update(
                **{
                    k: v
                    for k, v in original_snapshot.items()
                    if k not in ["last_updated"] and not k.startswith("_")
                }
            )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_crypto_rag(
    llm_client: Optional[Any] = None,
    retriever: Optional[Any] = None,
    top_k: int = CryptoRAG.DEFAULT_TOP_K,
) -> CryptoRAG:
    """
    Factory function to create a CryptoRAG instance.

    Args:
        llm_client: LLM client for generating responses
        retriever: Document retriever for message retrieval
        top_k: Number of messages to retrieve

    Returns:
        Configured CryptoRAG instance
    """
    return CryptoRAG(
        llm_client=llm_client,
        retriever=retriever,
        top_k=top_k,
    )


# =============================================================================
# OLLAMA INTEGRATION
# =============================================================================


class OllamaClient:
    """
    Simple Ollama client for local LLM inference.

    Wraps the Ollama API for use with CryptoRAG.
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (default: mistral)
            base_url: Ollama API base URL
        """
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """
        Generate a response using Ollama.

        Args:
            prompt: The prompt to send
            system: Optional system prompt
            temperature: Sampling temperature

        Returns:
            Response dictionary with 'text' key
        """
        import requests

        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return {"text": result.get("response", "")}

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return {"text": f"Error: {e}"}


def create_crypto_rag_with_ollama(
    model: str = "mistral",
    ollama_url: str = "http://localhost:11434",
    retriever: Optional[Any] = None,
    top_k: int = CryptoRAG.DEFAULT_TOP_K,
) -> CryptoRAG:
    """
    Create a CryptoRAG instance with Ollama as the LLM backend.

    Args:
        model: Ollama model name
        ollama_url: Ollama API URL
        retriever: Document retriever
        top_k: Number of messages to retrieve

    Returns:
        Configured CryptoRAG instance with Ollama

    Example:
        >>> rag = create_crypto_rag_with_ollama(
        ...     model="mistral",
        ...     ollama_url="http://ollama:11434",
        ...     retriever=doc_store,
        ... )
        >>> response = rag.answer("What's trending?")
    """
    ollama_client = OllamaClient(model=model, base_url=ollama_url)
    return CryptoRAG(
        llm_client=ollama_client,
        retriever=retriever,
        top_k=top_k,
    )
