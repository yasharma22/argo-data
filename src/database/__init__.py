"""
Database package for ARGO Data Platform

This package provides database models, connections, and operations
for storing and retrieving ARGO profile data.
"""

from .models import (
    Base,
    ArgoFloat,
    ArgoProfile,
    ArgoMeasurement,
    DataProcessingLog,
    VectorMetadata,
    QualityFlags
)

from .connection import (
    DatabaseManager,
    db_manager,
    get_db_session,
    get_db_context,
    init_database,
    test_db_connection,
    DatabaseError
)

__all__ = [
    'Base',
    'ArgoFloat',
    'ArgoProfile',
    'ArgoMeasurement',
    'DataProcessingLog',
    'VectorMetadata',
    'QualityFlags',
    'DatabaseManager',
    'db_manager',
    'get_db_session',
    'get_db_context',
    'init_database',
    'test_db_connection',
    'DatabaseError'
]
