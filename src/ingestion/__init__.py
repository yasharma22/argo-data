"""
ARGO Data Ingestion Module

This module provides functionality to ingest and process ARGO NetCDF files.
"""

from .argo_reader import ArgoNetCDFReader, ArgoProfile, process_argo_directory

__all__ = ['ArgoNetCDFReader', 'ArgoProfile', 'process_argo_directory']
