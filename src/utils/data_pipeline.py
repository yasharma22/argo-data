"""
Data Processing Pipeline

This module provides the end-to-end pipeline for processing ARGO NetCDF files
and populating both SQL and vector databases.
"""

import os
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from tqdm import tqdm
import numpy as np

from ..ingestion.argo_reader import ArgoNetCDFReader, ArgoProfile, process_argo_directory
from ..database.connection import get_db_context, DatabaseError
from ..database.models import (
    ArgoFloat, ArgoProfile as DBArgoProfile, ArgoMeasurement, 
    DataProcessingLog, VectorMetadata
)
from ..rag.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Main data processing pipeline"""
    
    def __init__(self):
        self.reader = ArgoNetCDFReader()
        self.vector_store = get_vector_store()
        
        # Parameter mappings from ARGO names to database column names
        self.param_mappings = {
            'TEMP': 'temperature',
            'PSAL': 'salinity', 
            'PRES': 'pressure',
            'DOXY': 'oxygen',
            'CHLA': 'chlorophyll',
            'BBP700': 'bbp700',
            'PH_IN_SITU_TOTAL': 'ph_total',
            'NITRATE': 'nitrate',
            'DOWNWELLING_PAR': 'downwelling_par'
        }
        
        self.qc_mappings = {
            'TEMP': 'temperature_qc',
            'PSAL': 'salinity_qc',
            'DOXY': 'oxygen_qc',
            'CHLA': 'chlorophyll_qc',
            'BBP700': 'bbp700_qc',
            'PH_IN_SITU_TOTAL': 'ph_total_qc',
            'NITRATE': 'nitrate_qc',
            'DOWNWELLING_PAR': 'downwelling_par_qc'
        }
    
    def process_single_file(self, file_path: str) -> Tuple[int, int, List[str]]:
        """
        Process a single NetCDF file
        
        Returns:
            Tuple of (successful_profiles, failed_profiles, error_messages)
        """
        
        operation_id = str(uuid.uuid4())
        errors = []
        
        try:
            # Create processing log entry
            with get_db_context() as session:
                log_entry = DataProcessingLog(
                    operation_id=operation_id,
                    operation_type='file_ingestion',
                    source_file=file_path,
                    status='running'
                )
                session.add(log_entry)
                session.flush()  # Get the ID
                
        except Exception as e:
            logger.error(f"Failed to create processing log: {e}")
            return 0, 1, [f"Failed to create processing log: {str(e)}"]
        
        try:
            # Read ARGO profiles from file
            profiles = self.reader.read_argo_file(file_path)
            
            if not profiles:
                error_msg = f"No profiles found in file: {file_path}"
                errors.append(error_msg)
                self._update_processing_log(operation_id, 'failed', error_message=error_msg)
                return 0, 1, errors
            
            successful_count = 0
            failed_count = 0
            
            # Process each profile
            for profile in profiles:
                try:
                    self._process_single_profile(profile)
                    successful_count += 1
                except Exception as e:
                    error_msg = f"Failed to process profile {profile.profile_id}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    failed_count += 1
            
            # Update processing log
            self._update_processing_log(
                operation_id, 
                'completed' if failed_count == 0 else 'partially_completed',
                profiles_processed=len(profiles),
                profiles_successful=successful_count,
                profiles_failed=failed_count,
                error_details={'errors': errors} if errors else None
            )
            
            logger.info(f"Processed file {file_path}: {successful_count} successful, {failed_count} failed")
            return successful_count, failed_count, errors
            
        except Exception as e:
            error_msg = f"Fatal error processing file {file_path}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            self._update_processing_log(operation_id, 'failed', error_message=error_msg)
            return 0, 1, errors
    
    def _process_single_profile(self, profile: ArgoProfile):
        """Process a single ARGO profile and store in databases"""
        
        with get_db_context() as session:
            # 1. Create or update ARGO float record
            existing_float = session.query(ArgoFloat).filter_by(
                platform_number=profile.platform_number
            ).first()
            
            if not existing_float:
                float_record = ArgoFloat(
                    platform_number=profile.platform_number,
                    project_name=profile.metadata.get('project_name'),
                    pi_name=profile.metadata.get('pi_name')
                )
                session.add(float_record)
            
            # 2. Create profile record
            existing_profile = session.query(DBArgoProfile).filter_by(
                profile_id=profile.profile_id
            ).first()
            
            if existing_profile:
                logger.warning(f"Profile {profile.profile_id} already exists, skipping")
                return
            
            db_profile = DBArgoProfile(
                profile_id=profile.profile_id,
                platform_number=profile.platform_number,
                cycle_number=profile.cycle_number,
                latitude=profile.latitude,
                longitude=profile.longitude,
                date=profile.date,
                depth_min=float(np.min(profile.depth_levels)),
                depth_max=float(np.max(profile.depth_levels)),
                n_levels=len(profile.depth_levels),
                data_mode=profile.metadata.get('data_mode'),
                direction=profile.metadata.get('direction'),
                parameters=list(profile.measurements.keys()),
                station_parameters=profile.metadata.get('station_parameters', []),
                metadata_json=profile.metadata
            )
            
            session.add(db_profile)
            session.flush()  # Ensure profile is created before measurements
            
            # 3. Create measurement records
            measurements = []
            
            # Get the common length (all parameters should have same length)
            if profile.measurements:
                param_name = list(profile.measurements.keys())[0]
                n_measurements = len(profile.measurements[param_name])
                
                for i in range(n_measurements):
                    measurement = ArgoMeasurement(
                        profile_id=profile.profile_id,
                        pressure=float(profile.depth_levels[i]) if i < len(profile.depth_levels) else None,
                        depth=float(profile.depth_levels[i]) if i < len(profile.depth_levels) else None  # Simplified: using pressure as depth
                    )
                    
                    # Add parameter values
                    for argo_param, values in profile.measurements.items():
                        if i < len(values) and argo_param in self.param_mappings:
                            db_column = self.param_mappings[argo_param]
                            value = float(values[i]) if not np.isnan(values[i]) else None
                            setattr(measurement, db_column, value)
                            
                            # Add quality control flags
                            if argo_param in profile.quality_flags and argo_param in self.qc_mappings:
                                qc_column = self.qc_mappings[argo_param]
                                qc_values = profile.quality_flags[argo_param]
                                if i < len(qc_values):
                                    qc_value = int(qc_values[i]) if not np.isnan(qc_values[i]) else None
                                    setattr(measurement, qc_column, qc_value)
                    
                    measurements.append(measurement)
            
            # Bulk insert measurements
            session.add_all(measurements)
            
            # 4. Create vector embeddings for the profile
            self._create_profile_embedding(profile, session)
            
            session.commit()
            logger.debug(f"Successfully processed profile {profile.profile_id}")
    
    def _create_profile_embedding(self, profile: ArgoProfile, session):
        """Create and store vector embeddings for the profile"""
        
        # Generate text summary for embedding
        summary_text = self._generate_profile_summary(profile)
        
        # Create metadata for the document
        metadata = {
            'profile_id': profile.profile_id,
            'platform_number': profile.platform_number,
            'document_type': 'profile_summary',
            'latitude': profile.latitude,
            'longitude': profile.longitude,
            'date': profile.date.isoformat() if profile.date else None,
            'parameters': list(profile.measurements.keys()),
            'depth_range': [float(np.min(profile.depth_levels)), float(np.max(profile.depth_levels))]
        }
        
        # Add to vector store
        try:
            doc_ids = self.vector_store.add_documents(
                texts=[summary_text],
                metadatas=[metadata],
                document_ids=[f"profile_{profile.profile_id}"]
            )
            
            logger.debug(f"Created vector embedding for profile {profile.profile_id}")
            
        except Exception as e:
            logger.error(f"Failed to create vector embedding for {profile.profile_id}: {e}")
    
    def _generate_profile_summary(self, profile: ArgoProfile) -> str:
        """Generate a text summary of the profile for embedding"""
        
        # Format date
        date_str = profile.date.strftime("%Y-%m-%d") if profile.date else "unknown date"
        
        # Determine region
        region = self._determine_region(profile.latitude, profile.longitude)
        
        # Get parameter list
        params = list(profile.measurements.keys())
        param_str = ", ".join(params) if params else "no parameters"
        
        # Depth info
        min_depth = np.min(profile.depth_levels)
        max_depth = np.max(profile.depth_levels)
        
        summary = f"""
ARGO Profile {profile.profile_id} from float {profile.platform_number}, cycle {profile.cycle_number}.
Collected on {date_str} at location {profile.latitude:.2f}°N, {profile.longitude:.2f}°E in the {region}.
Measurements from {min_depth:.1f}m to {max_depth:.1f}m depth ({len(profile.depth_levels)} levels).
Available parameters: {param_str}.
Data mode: {profile.metadata.get('data_mode', 'unknown')}.
Project: {profile.metadata.get('project_name', 'unknown')}.
        """.strip()
        
        return summary
    
    def _determine_region(self, lat: float, lon: float) -> str:
        """Determine ocean region based on coordinates"""
        
        # Normalize longitude to -180 to 180
        while lon > 180:
            lon -= 360
        while lon < -180:
            lon += 360
        
        # Define regions
        if lat >= 70:
            return "Arctic Ocean"
        elif lat <= -50:
            return "Southern Ocean"
        elif 30 <= lat <= 46 and -6 <= lon <= 42:
            return "Mediterranean Sea"
        elif 10 <= lat <= 30 and 50 <= lon <= 80:
            return "Arabian Sea"
        elif 5 <= lat <= 25 and 80 <= lon <= 100:
            return "Bay of Bengal"
        elif lat >= 20 and (120 <= lon <= 240 or -240 <= lon <= -120):
            return "North Pacific Ocean"
        elif lat < 0 and (120 <= lon <= 280 or -240 <= lon <= -80):
            return "South Pacific Ocean"
        elif lat >= 40 and -80 <= lon <= 0:
            return "North Atlantic Ocean"
        elif lat < 0 and -70 <= lon <= 20:
            return "South Atlantic Ocean"
        elif -60 <= lat <= 30 and 20 <= lon <= 120:
            return "Indian Ocean"
        else:
            return "Unknown Ocean"
    
    def _update_processing_log(self, operation_id: str, status: str, 
                             profiles_processed: int = None, profiles_successful: int = None,
                             profiles_failed: int = None, error_message: str = None,
                             error_details: dict = None):
        """Update the processing log entry"""
        
        try:
            with get_db_context() as session:
                log_entry = session.query(DataProcessingLog).filter_by(
                    operation_id=operation_id
                ).first()
                
                if log_entry:
                    log_entry.status = status
                    log_entry.end_time = datetime.utcnow()
                    
                    if profiles_processed is not None:
                        log_entry.profiles_processed = profiles_processed
                    if profiles_successful is not None:
                        log_entry.profiles_successful = profiles_successful
                    if profiles_failed is not None:
                        log_entry.profiles_failed = profiles_failed
                    if error_message:
                        log_entry.error_message = error_message
                    if error_details:
                        log_entry.error_details = error_details
                        
                    session.commit()
                
        except Exception as e:
            logger.error(f"Failed to update processing log: {e}")
    
    def process_directory(self, directory_path: str) -> Dict[str, any]:
        """
        Process all NetCDF files in a directory
        
        Returns:
            Summary dictionary with processing results
        """
        
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Find all NetCDF files
        netcdf_files = [
            os.path.join(directory_path, f) 
            for f in os.listdir(directory_path) 
            if f.endswith('.nc')
        ]
        
        if not netcdf_files:
            logger.warning(f"No NetCDF files found in {directory_path}")
            return {
                'total_files': 0,
                'successful_files': 0,
                'failed_files': 0,
                'total_profiles': 0,
                'successful_profiles': 0,
                'failed_profiles': 0,
                'errors': []
            }
        
        logger.info(f"Found {len(netcdf_files)} NetCDF files to process")
        
        # Process files with progress bar
        total_successful_profiles = 0
        total_failed_profiles = 0
        successful_files = 0
        failed_files = 0
        all_errors = []
        
        for file_path in tqdm(netcdf_files, desc="Processing files"):
            try:
                successful, failed, errors = self.process_single_file(file_path)
                
                total_successful_profiles += successful
                total_failed_profiles += failed
                all_errors.extend(errors)
                
                if successful > 0:
                    successful_files += 1
                if failed > 0:
                    failed_files += 1
                    
            except Exception as e:
                error_msg = f"Fatal error processing {file_path}: {str(e)}"
                all_errors.append(error_msg)
                logger.error(error_msg)
                failed_files += 1
        
        # Save vector store
        try:
            self.vector_store.save()
            logger.info("Vector store saved successfully")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
        
        summary = {
            'total_files': len(netcdf_files),
            'successful_files': successful_files,
            'failed_files': failed_files,
            'total_profiles': total_successful_profiles + total_failed_profiles,
            'successful_profiles': total_successful_profiles,
            'failed_profiles': total_failed_profiles,
            'errors': all_errors
        }
        
        logger.info(f"Processing complete: {summary}")
        return summary
    
    def get_processing_stats(self) -> Dict[str, any]:
        """Get processing statistics from the database"""
        
        try:
            with get_db_context() as session:
                # Count profiles and measurements
                profile_count = session.query(DBArgoProfile).count()
                measurement_count = session.query(ArgoMeasurement).count()
                float_count = session.query(ArgoFloat).count()
                
                # Get recent processing logs
                recent_logs = session.query(DataProcessingLog)\
                    .order_by(DataProcessingLog.start_time.desc())\
                    .limit(10)\
                    .all()
                
                # Get vector store stats
                vector_stats = self.vector_store.get_stats()
                
                return {
                    'database_stats': {
                        'floats': float_count,
                        'profiles': profile_count,
                        'measurements': measurement_count
                    },
                    'vector_store_stats': vector_stats,
                    'recent_processing': [
                        {
                            'operation_id': log.operation_id,
                            'operation_type': log.operation_type,
                            'status': log.status,
                            'start_time': log.start_time.isoformat(),
                            'profiles_processed': log.profiles_processed,
                            'profiles_successful': log.profiles_successful,
                            'profiles_failed': log.profiles_failed
                        }
                        for log in recent_logs
                    ]
                }
                
        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            return {'error': str(e)}


# Utility functions
def process_argo_files(directory_path: str) -> Dict[str, any]:
    """Convenience function to process ARGO files"""
    processor = DataProcessor()
    return processor.process_directory(directory_path)


def get_processing_statistics() -> Dict[str, any]:
    """Get processing statistics"""
    processor = DataProcessor()
    return processor.get_processing_stats()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python data_pipeline.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    print(f"Processing ARGO files in: {directory_path}")
    
    try:
        results = process_argo_files(directory_path)
        print(f"Processing completed:")
        print(f"  Files: {results['successful_files']}/{results['total_files']} successful")
        print(f"  Profiles: {results['successful_profiles']}/{results['total_profiles']} successful")
        
        if results['errors']:
            print(f"  Errors: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"    - {error}")
    
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
