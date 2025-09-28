"""
Database Models for ARGO Data Platform

This module defines the SQLAlchemy models for storing ARGO profile data
and related metadata in PostgreSQL.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime
import uuid

Base = declarative_base()


class ArgoFloat(Base):
    """Table to store ARGO float information"""
    __tablename__ = 'argo_floats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    platform_number = Column(String(50), unique=True, nullable=False, index=True)
    project_name = Column(String(200))
    pi_name = Column(String(200))
    wmo_inst_type = Column(String(50))
    data_centre = Column(String(10))
    date_creation = Column(DateTime, default=datetime.utcnow)
    date_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    profiles = relationship("ArgoProfile", back_populates="float", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ArgoFloat(platform_number='{self.platform_number}')>"


class ArgoProfile(Base):
    """Table to store ARGO profile metadata"""
    __tablename__ = 'argo_profiles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(String(100), unique=True, nullable=False, index=True)
    platform_number = Column(String(50), ForeignKey('argo_floats.platform_number'), nullable=False)
    cycle_number = Column(Integer, nullable=False)
    
    # Location and time
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    
    # Profile characteristics
    depth_min = Column(Float)
    depth_max = Column(Float)
    n_levels = Column(Integer)
    
    # Data quality and mode
    data_mode = Column(String(1))  # R (real-time), D (delayed-mode), A (adjusted)
    direction = Column(String(1))  # A (ascending), D (descending)
    
    # Available parameters
    parameters = Column(ARRAY(String))
    station_parameters = Column(ARRAY(String))
    
    # Metadata
    metadata_json = Column(JSON)
    
    # Timestamps
    date_creation = Column(DateTime, default=datetime.utcnow)
    date_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    float = relationship("ArgoFloat", back_populates="profiles")
    measurements = relationship("ArgoMeasurement", back_populates="profile", cascade="all, delete-orphan")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_profile_location', 'latitude', 'longitude'),
        Index('idx_profile_date_location', 'date', 'latitude', 'longitude'),
        Index('idx_profile_platform_cycle', 'platform_number', 'cycle_number'),
    )
    
    def __repr__(self):
        return f"<ArgoProfile(profile_id='{self.profile_id}')>"


class ArgoMeasurement(Base):
    """Table to store individual measurements from ARGO profiles"""
    __tablename__ = 'argo_measurements'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    profile_id = Column(String(100), ForeignKey('argo_profiles.profile_id'), nullable=False)
    
    # Measurement data
    pressure = Column(Float, nullable=False)  # dbar
    depth = Column(Float)  # meters (calculated from pressure)
    
    # Core parameters
    temperature = Column(Float)  # degrees Celsius
    salinity = Column(Float)  # PSU
    
    # BGC parameters
    oxygen = Column(Float)  # micromol/kg
    chlorophyll = Column(Float)  # mg/m3
    bbp700 = Column(Float)  # m-1
    ph_total = Column(Float)
    nitrate = Column(Float)  # micromol/kg
    downwelling_par = Column(Float)  # microMoleQuanta/m2/sec
    
    # Quality flags (stored as integers)
    temperature_qc = Column(Integer)
    salinity_qc = Column(Integer)
    oxygen_qc = Column(Integer)
    chlorophyll_qc = Column(Integer)
    bbp700_qc = Column(Integer)
    ph_total_qc = Column(Integer)
    nitrate_qc = Column(Integer)
    downwelling_par_qc = Column(Integer)
    
    # Timestamp
    date_creation = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    profile = relationship("ArgoProfile", back_populates="measurements")
    
    # Indexes for efficient queries
    __table_args__ = (
        Index('idx_measurement_profile', 'profile_id'),
        Index('idx_measurement_pressure', 'pressure'),
        Index('idx_measurement_temp_sal', 'temperature', 'salinity'),
        Index('idx_measurement_bgc', 'oxygen', 'chlorophyll'),
    )
    
    def __repr__(self):
        return f"<ArgoMeasurement(profile_id='{self.profile_id}', pressure={self.pressure})>"


class DataProcessingLog(Base):
    """Table to track data processing operations"""
    __tablename__ = 'data_processing_log'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    operation_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True)
    operation_type = Column(String(50), nullable=False)  # 'file_ingestion', 'profile_processing', etc.
    
    # File/data information
    source_file = Column(String(500))
    profiles_processed = Column(Integer, default=0)
    profiles_successful = Column(Integer, default=0)
    profiles_failed = Column(Integer, default=0)
    
    # Status and timing
    status = Column(String(20), default='running')  # 'running', 'completed', 'failed'
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    
    # Error information
    error_message = Column(Text)
    error_details = Column(JSON)
    
    # Additional metadata
    processing_metadata = Column(JSON)
    
    def __repr__(self):
        return f"<DataProcessingLog(operation_id='{self.operation_id}', status='{self.status}')>"


class VectorMetadata(Base):
    """Table to store vector database metadata for RAG system"""
    __tablename__ = 'vector_metadata'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vector_id = Column(String(100), unique=True, nullable=False, index=True)
    document_type = Column(String(50), nullable=False)  # 'profile_summary', 'measurement_summary', etc.
    
    # Related data identifiers
    profile_id = Column(String(100), ForeignKey('argo_profiles.profile_id'))
    platform_number = Column(String(50))
    
    # Vector information
    embedding_model = Column(String(100))
    vector_dimension = Column(Integer)
    
    # Content information
    content_summary = Column(Text)
    content_metadata = Column(JSON)
    
    # Timestamps
    date_creation = Column(DateTime, default=datetime.utcnow)
    date_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Index for efficient queries
    __table_args__ = (
        Index('idx_vector_document_type', 'document_type'),
        Index('idx_vector_profile', 'profile_id'),
    )
    
    def __repr__(self):
        return f"<VectorMetadata(vector_id='{self.vector_id}', document_type='{self.document_type}')>"


# Quality control lookup tables
class QualityFlags(Base):
    """Reference table for ARGO quality flag meanings"""
    __tablename__ = 'quality_flags'
    
    flag_value = Column(Integer, primary_key=True)
    flag_meaning = Column(String(100), nullable=False)
    flag_description = Column(Text)
    
    def __repr__(self):
        return f"<QualityFlags(flag_value={self.flag_value}, meaning='{self.flag_meaning}')>"


# Initialize quality flags with ARGO standard values
def initialize_quality_flags():
    """Initialize the quality flags table with ARGO standard values"""
    return [
        QualityFlags(flag_value=0, flag_meaning="no_qc_performed", 
                    flag_description="No quality control tests performed"),
        QualityFlags(flag_value=1, flag_meaning="good_data", 
                    flag_description="Passed all quality control tests"),
        QualityFlags(flag_value=2, flag_meaning="probably_good_data", 
                    flag_description="Passed non-real-time quality control tests"),
        QualityFlags(flag_value=3, flag_meaning="bad_data_correctable", 
                    flag_description="Failed non-real-time quality control tests that are correctable"),
        QualityFlags(flag_value=4, flag_meaning="bad_data", 
                    flag_description="Failed quality control tests"),
        QualityFlags(flag_value=5, flag_meaning="value_changed", 
                    flag_description="Value changed"),
        QualityFlags(flag_value=6, flag_meaning="not_used", 
                    flag_description="Not used"),
        QualityFlags(flag_value=7, flag_meaning="not_used", 
                    flag_description="Not used"),
        QualityFlags(flag_value=8, flag_meaning="interpolated_value", 
                    flag_description="Interpolated value"),
        QualityFlags(flag_value=9, flag_meaning="missing_value", 
                    flag_description="Missing value")
    ]
