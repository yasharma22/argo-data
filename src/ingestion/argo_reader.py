"""
ARGO NetCDF Data Reader and Parser

This module provides functionality to read and parse ARGO NetCDF files,
extracting profile data and metadata for further processing.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArgoProfile:
    """Data class to represent an ARGO profile"""
    profile_id: str
    platform_number: str
    cycle_number: int
    latitude: float
    longitude: float
    date: datetime
    measurements: Dict[str, np.ndarray]  # parameter_name -> values
    depth_levels: np.ndarray
    quality_flags: Dict[str, np.ndarray]
    metadata: Dict[str, any]


class ArgoNetCDFReader:
    """Class to read and parse ARGO NetCDF files"""
    
    def __init__(self):
        self.supported_parameters = [
            'TEMP', 'PSAL', 'PRES',  # Core parameters
            'DOXY', 'CHLA', 'BBP700', 'PH_IN_SITU_TOTAL',  # BGC parameters
            'NITRATE', 'DOWNWELLING_PAR'
        ]
    
    def read_argo_file(self, file_path: str) -> List[ArgoProfile]:
        """
        Read an ARGO NetCDF file and extract profiles
        
        Args:
            file_path: Path to the NetCDF file
            
        Returns:
            List of ArgoProfile objects
        """
        try:
            with xr.open_dataset(file_path) as ds:
                profiles = []
                n_profiles = ds.dims.get('N_PROF', 1)
                
                for prof_idx in range(n_profiles):
                    profile = self._extract_profile(ds, prof_idx)
                    if profile:
                        profiles.append(profile)
                
                logger.info(f"Successfully read {len(profiles)} profiles from {file_path}")
                return profiles
                
        except Exception as e:
            logger.error(f"Error reading ARGO file {file_path}: {str(e)}")
            return []
    
    def _extract_profile(self, ds: xr.Dataset, prof_idx: int) -> Optional[ArgoProfile]:
        """Extract a single profile from the dataset"""
        try:
            # Extract basic metadata
            platform_number = self._get_string_variable(ds, 'PLATFORM_NUMBER', prof_idx)
            cycle_number = int(ds['CYCLE_NUMBER'].values[prof_idx])
            
            # Extract position and time
            latitude = float(ds['LATITUDE'].values[prof_idx])
            longitude = float(ds['LONGITUDE'].values[prof_idx])
            
            # Handle time conversion
            reference_date = ds['REFERENCE_DATE_TIME'].values[prof_idx]
            juld = ds['JULD'].values[prof_idx]
            date = self._convert_argo_time(reference_date, juld)
            
            if not date:
                return None
            
            # Extract measurements
            measurements = {}
            quality_flags = {}
            
            # Get depth/pressure levels
            if 'PRES' in ds.variables:
                depth_levels = ds['PRES'].values[prof_idx, :]
                depth_levels = depth_levels[~np.isnan(depth_levels)]
            else:
                return None
            
            # Extract available parameters
            for param in self.supported_parameters:
                if param in ds.variables:
                    values = ds[param].values[prof_idx, :]
                    # Remove NaN values and align with depth
                    valid_mask = ~np.isnan(values) & ~np.isnan(depth_levels)
                    if np.any(valid_mask):
                        measurements[param] = values[valid_mask]
                        
                        # Extract quality flags if available
                        qc_var = f"{param}_QC"
                        if qc_var in ds.variables:
                            quality_flags[param] = ds[qc_var].values[prof_idx, :][valid_mask]
            
            # Create profile ID
            profile_id = f"{platform_number}_{cycle_number}"
            
            # Extract additional metadata
            metadata = {
                'data_mode': self._get_string_variable(ds, 'DATA_MODE', prof_idx),
                'direction': self._get_string_variable(ds, 'DIRECTION', prof_idx),
                'station_parameters': self._get_station_parameters(ds, prof_idx),
                'pi_name': self._get_string_variable(ds, 'PI_NAME', prof_idx) if 'PI_NAME' in ds.variables else None,
                'project_name': self._get_string_variable(ds, 'PROJECT_NAME', prof_idx) if 'PROJECT_NAME' in ds.variables else None,
            }
            
            return ArgoProfile(
                profile_id=profile_id,
                platform_number=platform_number,
                cycle_number=cycle_number,
                latitude=latitude,
                longitude=longitude,
                date=date,
                measurements=measurements,
                depth_levels=depth_levels[valid_mask] if 'valid_mask' in locals() else depth_levels,
                quality_flags=quality_flags,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error extracting profile {prof_idx}: {str(e)}")
            return None
    
    def _get_string_variable(self, ds: xr.Dataset, var_name: str, prof_idx: int) -> str:
        """Extract string variable from NetCDF dataset"""
        if var_name not in ds.variables:
            return ""
        
        try:
            value = ds[var_name].values[prof_idx]
            if isinstance(value, bytes):
                return value.decode('utf-8').strip()
            elif isinstance(value, str):
                return value.strip()
            else:
                return str(value).strip()
        except:
            return ""
    
    def _get_station_parameters(self, ds: xr.Dataset, prof_idx: int) -> List[str]:
        """Extract station parameters list"""
        if 'STATION_PARAMETERS' not in ds.variables:
            return []
        
        try:
            params = ds['STATION_PARAMETERS'].values[prof_idx]
            if len(params.shape) > 1:
                # Multiple parameters
                return [p.decode('utf-8').strip() if isinstance(p, bytes) else str(p).strip() 
                       for p in params.flatten() if p]
            else:
                # Single parameter
                param = params.item() if hasattr(params, 'item') else params
                return [param.decode('utf-8').strip() if isinstance(param, bytes) else str(param).strip()]
        except:
            return []
    
    def _convert_argo_time(self, reference_date: str, juld: float) -> Optional[datetime]:
        """Convert ARGO time format to datetime"""
        try:
            if np.isnan(juld):
                return None
            
            # Reference date is typically '19500101000000'
            ref_str = reference_date.decode('utf-8').strip() if isinstance(reference_date, bytes) else str(reference_date).strip()
            ref_dt = datetime.strptime(ref_str, '%Y%m%d%H%M%S')
            
            # JULD is days since reference date
            from datetime import timedelta
            return ref_dt + timedelta(days=float(juld))
            
        except Exception as e:
            logger.warning(f"Could not convert ARGO time: {str(e)}")
            return None
    
    def get_profile_summary(self, profile: ArgoProfile) -> Dict:
        """Generate a summary of the profile for metadata storage"""
        return {
            'profile_id': profile.profile_id,
            'platform_number': profile.platform_number,
            'cycle_number': profile.cycle_number,
            'latitude': profile.latitude,
            'longitude': profile.longitude,
            'date': profile.date.isoformat() if profile.date else None,
            'parameters': list(profile.measurements.keys()),
            'depth_range': [float(np.min(profile.depth_levels)), float(np.max(profile.depth_levels))],
            'n_measurements': len(profile.depth_levels),
            'data_mode': profile.metadata.get('data_mode', ''),
            'project': profile.metadata.get('project_name', ''),
        }


def process_argo_directory(directory_path: str) -> List[ArgoProfile]:
    """
    Process all ARGO NetCDF files in a directory
    
    Args:
        directory_path: Path to directory containing ARGO files
        
    Returns:
        List of all ArgoProfile objects
    """
    reader = ArgoNetCDFReader()
    all_profiles = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.nc'):
            file_path = os.path.join(directory_path, filename)
            profiles = reader.read_argo_file(file_path)
            all_profiles.extend(profiles)
    
    logger.info(f"Processed {len(all_profiles)} total profiles from {directory_path}")
    return all_profiles


if __name__ == "__main__":
    # Example usage
    reader = ArgoNetCDFReader()
    # profiles = reader.read_argo_file("path/to/argo/file.nc")
    # for profile in profiles:
    #     print(reader.get_profile_summary(profile))
