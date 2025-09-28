"""
Sample ARGO Data Generator

This module creates sample NetCDF files that mimic real ARGO data
for testing and demonstration purposes.
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleArgoGenerator:
    """Generate sample ARGO NetCDF files for testing"""
    
    def __init__(self, output_dir: str = "./data/sample_argo"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Standard ARGO parameters
        self.core_parameters = ['TEMP', 'PSAL', 'PRES']
        self.bgc_parameters = ['DOXY', 'CHLA', 'BBP700', 'PH_IN_SITU_TOTAL', 'NITRATE']
        
        # Quality flag meanings (ARGO standard)
        self.qc_flags = {
            0: "no_qc_performed",
            1: "good_data", 
            2: "probably_good_data",
            3: "bad_data_correctable",
            4: "bad_data",
            9: "missing_value"
        }
    
    def generate_realistic_profile(self, platform_number: str, cycle_number: int, 
                                 lat: float, lon: float, date: datetime,
                                 include_bgc: bool = False) -> xr.Dataset:
        """Generate a realistic ARGO profile dataset"""
        
        # Generate depth levels (pressure)
        n_levels = np.random.randint(50, 150)
        pressure_levels = np.sort(np.random.uniform(5, 2000, n_levels))
        
        # Create coordinate arrays
        n_prof = 1  # Single profile
        
        # Basic metadata
        dataset = xr.Dataset(
            coords={
                'N_PROF': np.arange(n_prof),
                'N_LEVELS': np.arange(n_levels)
            }
        )
        
        # Platform and cycle information
        dataset['PLATFORM_NUMBER'] = (['N_PROF'], [platform_number.encode('utf-8')])
        dataset['CYCLE_NUMBER'] = (['N_PROF'], [cycle_number])
        
        # Location and time
        dataset['LATITUDE'] = (['N_PROF'], [lat])
        dataset['LONGITUDE'] = (['N_PROF'], [lon])
        
        # Time handling - ARGO uses Julian days since reference
        reference_date = "19500101000000"  # ARGO reference
        ref_datetime = datetime.strptime(reference_date, "%Y%m%d%H%M%S")
        days_since_ref = (date - ref_datetime).days
        
        dataset['REFERENCE_DATE_TIME'] = (['N_PROF'], [reference_date.encode('utf-8')])
        dataset['JULD'] = (['N_PROF'], [days_since_ref])
        
        # Data mode and direction
        dataset['DATA_MODE'] = (['N_PROF'], ['R'])  # Real-time
        dataset['DIRECTION'] = (['N_PROF'], ['A'])  # Ascending
        
        # Pressure (depth) levels
        pres_data = np.full((n_prof, n_levels), np.nan)
        pres_data[0, :len(pressure_levels)] = pressure_levels
        dataset['PRES'] = (['N_PROF', 'N_LEVELS'], pres_data)
        
        # Generate core parameters
        temp_data, temp_qc = self._generate_temperature_profile(pressure_levels, lat)
        sal_data, sal_qc = self._generate_salinity_profile(pressure_levels, lat, lon)
        
        # Add to dataset with proper dimensions
        temp_full = np.full((n_prof, n_levels), np.nan)
        sal_full = np.full((n_prof, n_levels), np.nan)
        temp_qc_full = np.full((n_prof, n_levels), 9)  # Missing by default
        sal_qc_full = np.full((n_prof, n_levels), 9)
        
        temp_full[0, :len(temp_data)] = temp_data
        sal_full[0, :len(sal_data)] = sal_data
        temp_qc_full[0, :len(temp_qc)] = temp_qc
        sal_qc_full[0, :len(sal_qc)] = sal_qc
        
        dataset['TEMP'] = (['N_PROF', 'N_LEVELS'], temp_full)
        dataset['PSAL'] = (['N_PROF', 'N_LEVELS'], sal_full)
        dataset['TEMP_QC'] = (['N_PROF', 'N_LEVELS'], temp_qc_full)
        dataset['PSAL_QC'] = (['N_PROF', 'N_LEVELS'], sal_qc_full)
        
        # Add BGC parameters if requested
        if include_bgc:
            oxy_data, oxy_qc = self._generate_oxygen_profile(pressure_levels, temp_data)
            chla_data, chla_qc = self._generate_chlorophyll_profile(pressure_levels)
            
            oxy_full = np.full((n_prof, n_levels), np.nan)
            chla_full = np.full((n_prof, n_levels), np.nan)
            oxy_qc_full = np.full((n_prof, n_levels), 9)
            chla_qc_full = np.full((n_prof, n_levels), 9)
            
            oxy_full[0, :len(oxy_data)] = oxy_data
            chla_full[0, :len(chla_data)] = chla_data
            oxy_qc_full[0, :len(oxy_qc)] = oxy_qc
            chla_qc_full[0, :len(chla_qc)] = chla_qc
            
            dataset['DOXY'] = (['N_PROF', 'N_LEVELS'], oxy_full)
            dataset['CHLA'] = (['N_PROF', 'N_LEVELS'], chla_full)
            dataset['DOXY_QC'] = (['N_PROF', 'N_LEVELS'], oxy_qc_full)
            dataset['CHLA_QC'] = (['N_PROF', 'N_LEVELS'], chla_qc_full)
        
        # Station parameters
        parameters = self.core_parameters.copy()
        if include_bgc:
            parameters.extend(['DOXY', 'CHLA'])
        
        # Convert parameters to fixed-length strings for NetCDF
        param_array = np.array([p.ljust(8)[:8] for p in parameters])
        dataset['STATION_PARAMETERS'] = (['N_PROF', 'N_PARAM'], 
                                        param_array.reshape(1, -1))
        
        # Add global attributes
        dataset.attrs.update({
            'title': 'Sample ARGO profile data',
            'institution': 'Test Data Generator',
            'source': 'Generated sample data for ARGO platform testing',
            'date_created': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'Conventions': 'Argo-3.1 CF-1.6',
            'format_version': '3.1'
        })
        
        return dataset
    
    def _generate_temperature_profile(self, pressure: np.ndarray, lat: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic temperature profile"""
        
        # Base temperature depends on latitude
        if abs(lat) < 30:  # Tropical
            surface_temp = np.random.uniform(24, 30)
            deep_temp = np.random.uniform(2, 4)
        elif abs(lat) < 60:  # Temperate
            surface_temp = np.random.uniform(10, 20)
            deep_temp = np.random.uniform(1, 3)
        else:  # Polar
            surface_temp = np.random.uniform(-2, 8)
            deep_temp = np.random.uniform(-1, 2)
        
        # Exponential decay with depth
        temperature = surface_temp * np.exp(-pressure / 1000) + deep_temp
        
        # Add realistic variability
        temperature += np.random.normal(0, 0.3, len(temperature))
        
        # Ensure physical limits
        temperature = np.maximum(temperature, -2.0)  # Freezing point of seawater
        temperature = np.minimum(temperature, 35.0)  # Maximum realistic
        
        # Generate quality flags (mostly good data)
        qc_flags = np.random.choice([1, 2], size=len(temperature), p=[0.9, 0.1])
        
        # Add some bad data occasionally
        bad_indices = np.random.choice(len(temperature), 
                                     size=int(0.02 * len(temperature)), 
                                     replace=False)
        qc_flags[bad_indices] = 4
        temperature[bad_indices] = np.nan
        
        return temperature, qc_flags
    
    def _generate_salinity_profile(self, pressure: np.ndarray, lat: float, lon: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic salinity profile"""
        
        # Base salinity depends on region
        if abs(lat) < 30:  # Tropical - higher salinity
            base_salinity = np.random.uniform(35.5, 36.5)
        elif abs(lat) < 60:  # Temperate
            base_salinity = np.random.uniform(34.5, 35.5)
        else:  # Polar - fresher
            base_salinity = np.random.uniform(33.5, 34.5)
        
        # Mediterranean and Red Sea are saltier
        if (30 < lat < 46 and -6 < lon < 42) or (10 < lat < 30 and 32 < lon < 44):
            base_salinity += np.random.uniform(1, 2)
        
        # Slight increase with depth in most regions
        salinity = base_salinity + pressure * 0.0001
        
        # Add halocline structure
        halocline_depth = np.random.uniform(100, 500)
        salinity_jump = np.random.uniform(0.2, 0.8)
        salinity += salinity_jump / (1 + np.exp(-(pressure - halocline_depth) / 50))
        
        # Add variability
        salinity += np.random.normal(0, 0.1, len(salinity))
        
        # Physical limits
        salinity = np.maximum(salinity, 30.0)
        salinity = np.minimum(salinity, 42.0)
        
        # Quality flags
        qc_flags = np.random.choice([1, 2], size=len(salinity), p=[0.85, 0.15])
        
        # Add some bad data
        bad_indices = np.random.choice(len(salinity), 
                                     size=int(0.03 * len(salinity)), 
                                     replace=False)
        qc_flags[bad_indices] = 4
        salinity[bad_indices] = np.nan
        
        return salinity, qc_flags
    
    def _generate_oxygen_profile(self, pressure: np.ndarray, temperature: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic oxygen profile"""
        
        # Surface saturation depends on temperature
        surface_saturation = 400 - 10 * temperature[0]  # Rough approximation
        surface_oxygen = np.random.uniform(0.8, 1.0) * surface_saturation
        
        # Oxygen minimum zone
        omz_depth = np.random.uniform(400, 1000)
        min_oxygen = np.random.uniform(10, 80)
        
        # Deep water recovery
        deep_oxygen = np.random.uniform(150, 250)
        
        # Create oxygen profile with OMZ
        oxygen = np.zeros_like(pressure)
        for i, p in enumerate(pressure):
            if p < 100:  # Surface layer
                oxygen[i] = surface_oxygen * (1 - p / 200)
            elif p < omz_depth:  # Declining to OMZ
                decay_factor = (p - 100) / (omz_depth - 100)
                oxygen[i] = surface_oxygen * (1 - decay_factor) + min_oxygen * decay_factor
            elif p < omz_depth + 500:  # OMZ
                oxygen[i] = min_oxygen
            else:  # Deep water recovery
                recovery_factor = min(1, (p - omz_depth - 500) / 1000)
                oxygen[i] = min_oxygen + (deep_oxygen - min_oxygen) * recovery_factor
        
        # Add noise
        oxygen += np.random.normal(0, 5, len(oxygen))
        oxygen = np.maximum(oxygen, 0)  # Can't be negative
        
        # Quality flags (BGC data often has more issues)
        qc_flags = np.random.choice([1, 2, 3], size=len(oxygen), p=[0.7, 0.2, 0.1])
        
        return oxygen, qc_flags
    
    def _generate_chlorophyll_profile(self, pressure: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic chlorophyll profile"""
        
        # Surface chlorophyll
        surface_chla = np.random.exponential(0.5)
        
        # Deep chlorophyll maximum
        dcm_depth = np.random.uniform(50, 150)
        dcm_intensity = surface_chla * np.random.uniform(1.5, 3.0)
        
        # Generate profile with DCM
        chlorophyll = np.zeros_like(pressure)
        for i, p in enumerate(pressure):
            if p < dcm_depth:
                # Increase to DCM
                factor = np.exp(-(p - dcm_depth)**2 / (2 * 30**2))
                chlorophyll[i] = surface_chla + (dcm_intensity - surface_chla) * factor
            else:
                # Exponential decay below DCM
                chlorophyll[i] = dcm_intensity * np.exp(-(p - dcm_depth) / 200)
        
        # Add noise
        chlorophyll += np.random.exponential(0.05, len(chlorophyll))
        chlorophyll = np.maximum(chlorophyll, 0.001)  # Small minimum
        
        # Quality flags
        qc_flags = np.random.choice([1, 2, 3], size=len(chlorophyll), p=[0.6, 0.25, 0.15])
        
        return chlorophyll, qc_flags
    
    def generate_sample_files(self, n_files: int = 50, include_bgc_ratio: float = 0.3) -> List[str]:
        """Generate multiple sample ARGO files"""
        
        generated_files = []
        
        for i in range(n_files):
            # Random platform number
            platform = f"590{np.random.randint(1000, 9999)}"
            cycle = np.random.randint(1, 200)
            
            # Random global location
            lat = np.random.uniform(-60, 70)
            lon = np.random.uniform(-180, 180)
            
            # Random date in last year
            days_ago = np.random.randint(0, 365)
            profile_date = datetime.now() - timedelta(days=days_ago)
            
            # Decide if BGC
            include_bgc = np.random.random() < include_bgc_ratio
            
            # Generate dataset
            dataset = self.generate_realistic_profile(
                platform, cycle, lat, lon, profile_date, include_bgc
            )
            
            # Create filename
            date_str = profile_date.strftime("%Y%m%d")
            bgc_str = "_Bgc" if include_bgc else ""
            filename = f"R{platform}_{cycle:03d}{bgc_str}_{date_str}.nc"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save to NetCDF
            try:
                dataset.to_netcdf(filepath, format='NETCDF4')
                generated_files.append(filepath)
                logger.info(f"Generated: {filename}")
            except Exception as e:
                logger.error(f"Failed to save {filename}: {e}")
            
            dataset.close()
        
        logger.info(f"Generated {len(generated_files)} sample ARGO files in {self.output_dir}")
        return generated_files


def create_sample_dataset(output_dir: str = "./data/sample_argo", 
                         n_files: int = 20, 
                         bgc_ratio: float = 0.3) -> List[str]:
    """Convenience function to create sample dataset"""
    
    generator = SampleArgoGenerator(output_dir)
    return generator.generate_sample_files(n_files, bgc_ratio)


if __name__ == "__main__":
    import sys
    
    # Command line usage
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./data/sample_argo"
    
    print(f"Generating {n_files} sample ARGO files in {output_dir}")
    
    files = create_sample_dataset(output_dir, n_files)
    print(f"Successfully generated {len(files)} files:")
    for f in files[:5]:  # Show first 5
        print(f"  - {os.path.basename(f)}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more files")
