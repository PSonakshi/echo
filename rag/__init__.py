"""
RAG (Retrieval-Augmented Generation) module for the Crypto Narrative Pulse Tracker.

This module provides:
- live_metrics: Shared state management for real-time metrics
- crypto_rag: Context-enriched RAG question answering

Requirements: 8.1, 8.2, 8.3, 8.4
"""

from rag.crypto_rag import (
    CryptoRAG,
    OllamaClient,
    RAGResponse,
    create_crypto_rag,
    create_crypto_rag_with_ollama,
)
from rag.live_metrics import (
    LiveMetrics,
    get_live_metrics,
    subscribe_to_pipeline_outputs,
    update_divergence_status,
    update_influencer_consensus,
    update_pulse_score,
    update_trending_phrases,
)

__all__ = [
    # Live Metrics
    "LiveMetrics",
    "get_live_metrics",
    "update_pulse_score",
    "update_trending_phrases",
    "update_influencer_consensus",
    "update_divergence_status",
    "subscribe_to_pipeline_outputs",
    # Crypto RAG
    "CryptoRAG",
    "RAGResponse",
    "OllamaClient",
    "create_crypto_rag",
    "create_crypto_rag_with_ollama",
]
