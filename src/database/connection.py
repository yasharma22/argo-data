"""
Database Connection and Session Management

This module handles database connections, session management, and database initialization
for the ARGO data platform.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from typing import Generator, Optional
import logging
from dotenv import load_dotenv

from .models import Base, QualityFlags, initialize_quality_flags

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection"""
        # Get database configuration from environment
        db_host = os.getenv('POSTGRES_HOST', 'localhost')
        db_port = os.getenv('POSTGRES_PORT', '5432')
        db_name = os.getenv('POSTGRES_DB', 'argo_data')
        db_user = os.getenv('POSTGRES_USER', 'postgres')
        db_password = os.getenv('POSTGRES_PASSWORD', '')
        
        # Create connection string
        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                connection_string,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=os.getenv('SQL_ECHO', 'false').lower() == 'true'
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info(f"Database connection initialized successfully to {db_host}:{db_port}/{db_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {str(e)}")
            raise
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            
            # Initialize quality flags if table is empty
            self._initialize_reference_data()
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {str(e)}")
            raise
    
    def _initialize_reference_data(self):
        """Initialize reference data tables"""
        with self.get_session() as session:
            try:
                # Check if quality flags are already initialized
                existing_flags = session.query(QualityFlags).count()
                
                if existing_flags == 0:
                    # Initialize quality flags
                    flags = initialize_quality_flags()
                    session.add_all(flags)
                    session.commit()
                    logger.info("Quality flags initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize reference data: {str(e)}")
                session.rollback()
                raise
    
    def get_session(self) -> Session:
        """Get a new database session"""
        if not self.SessionLocal:
            raise RuntimeError("Database connection not initialized")
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Context manager for database sessions with automatic cleanup"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {str(e)}")
            raise
    
    def get_engine(self):
        """Get the database engine"""
        return self.engine


# Global database manager instance
db_manager = DatabaseManager()


def get_db_session() -> Session:
    """Get a database session (for FastAPI dependency injection)"""
    return db_manager.get_session()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Get a database session with context management"""
    with db_manager.session_scope() as session:
        yield session


def init_database():
    """Initialize the database (create tables and reference data)"""
    db_manager.create_tables()


def test_db_connection() -> bool:
    """Test the database connection"""
    return db_manager.test_connection()


class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass


def execute_raw_sql(query: str, params: Optional[dict] = None) -> list:
    """Execute raw SQL query and return results"""
    try:
        with db_manager.session_scope() as session:
            result = session.execute(text(query), params or {})
            return result.fetchall()
    except SQLAlchemyError as e:
        logger.error(f"Raw SQL execution failed: {str(e)}")
        raise DatabaseError(f"SQL execution failed: {str(e)}")


def get_table_info(table_name: str) -> dict:
    """Get information about a database table"""
    try:
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = :table_name
        ORDER BY ordinal_position
        """
        
        with db_manager.session_scope() as session:
            result = session.execute(text(query), {"table_name": table_name})
            columns = result.fetchall()
            
            # Get table size
            size_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            size_result = session.execute(text(size_query))
            row_count = size_result.scalar()
            
            return {
                "table_name": table_name,
                "columns": [dict(row._mapping) for row in columns],
                "row_count": row_count
            }
            
    except Exception as e:
        logger.error(f"Failed to get table info for {table_name}: {str(e)}")
        raise DatabaseError(f"Failed to get table info: {str(e)}")


if __name__ == "__main__":
    # Test database connection and initialization
    print("Testing database connection...")
    if test_db_connection():
        print("✓ Database connection successful")
        print("Initializing database tables...")
        init_database()
        print("✓ Database initialization complete")
    else:
        print("✗ Database connection failed")
