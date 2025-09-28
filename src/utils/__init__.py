"""
Utilities package for ARGO Data Platform

This package provides utility functions and the main data processing pipeline.
"""

from .data_pipeline import (
    DataProcessor,
    process_argo_files,
    get_processing_statistics
)

__all__ = [
    'DataProcessor',
    'process_argo_files',
    'get_processing_statistics'
]
