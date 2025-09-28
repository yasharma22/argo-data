"""
RAG (Retrieval-Augmented Generation) package for ARGO Data Platform

This package provides functionality for natural language to SQL translation
using vector similarity search and large language models.
"""

from .vector_store import (
    VectorStore,
    EmbeddingModel,
    Document,
    get_vector_store,
    initialize_vector_store
)

from .llm_interface import (
    QueryProcessor,
    LLMInterface,
    QueryContext,
    get_query_processor,
    process_user_query
)

__all__ = [
    'VectorStore',
    'EmbeddingModel',
    'Document',
    'get_vector_store',
    'initialize_vector_store',
    'QueryProcessor',
    'LLMInterface',
    'QueryContext',
    'get_query_processor',
    'process_user_query'
]
