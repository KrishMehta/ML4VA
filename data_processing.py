import pandas as pd
import numpy as np
from census import Census
import os
from typing import Tuple, Dict, List
import geopandas as gpd
from shapely.geometry import Point
import requests
import json

class DataProcessor:
    def __init__(self, cdc_data_path: str, census_api_key: str):
        self.cdc_data_path = cdc_data_path
        self.census_api_key = census_api_key
        self.c = Census(census_api_key)
        
    def load_cdc_data(self) -> pd.DataFrame:
        """Load and preprocess CDC mortality data."""
        if not os.path.exists(self.cdc_data_path):
            raise FileNotFoundError(f"File not found: {self.cdc_data_path}")
        
        df = pd.read_csv(self.cdc_data_path)
        va = df[(df.ST_NAME == 'Virginia') & (df.Intent == 'All_Suicide')]
        return va
    
    def get_acs_data(self, year: int = 2022) -> pd.DataFrame:
        """Fetch ACS socio-demographic data for Virginia tracts."""
        # Get income data
        income_data = self.c.acs5.state_county_tract(
            ('B19013_001E', 'B19013_001M'),  # Median household income
            '51',  # Virginia FIPS code
            '*',  # All counties
            '*',  # All tracts
            year=year
        )
        
        # Get education data
        education_data = self.c.acs5.state_county_tract(
            ('B15003_022E', 'B15003_023E', 'B15003_024E', 'B15003_025E'),  # Bachelor's degree or higher
            '51',
            '*',
            '*',
            year=year
        )
        
        # Convert to DataFrame
        income_df = pd.DataFrame(income_data)
        education_df = pd.DataFrame(education_data)
        
        # Merge data
        acs_data = pd.merge(income_df, education_df, on=['state', 'county', 'tract'])
        
        # Create GEOID
        acs_data['GEOID'] = acs_data['state'] + acs_data['county'] + acs_data['tract']
        
        return acs_data
    
    def get_svi_data(self) -> pd.DataFrame:
        """Fetch Social Vulnerability Index data."""
        # SVI data URL (CDC's SVI database)
        svi_url = "https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html"
        
        # Note: In practice, you would need to download and process the SVI data
        # This is a placeholder for the actual implementation
        return pd.DataFrame()
    
    def create_spatial_features(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Create spatial features using tract geometries."""
        # Load Virginia tract boundaries
        va_tracts = gpd.read_file("https://raw.githubusercontent.com/OpenDataIS/virginia-census-tracts/master/virginia-census-tracts.geojson")
        
        # Merge with our data
        spatial_df = pd.merge(df, va_tracts, on='GEOID', how='left')
        
        # Calculate spatial features
        spatial_df['centroid'] = spatial_df.geometry.centroid
        spatial_df['area'] = spatial_df.geometry.area
        
        return spatial_df
    
    def prepare_features(self, df: pd.DataFrame, acs_data: pd.DataFrame, svi_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare final feature matrix and labels."""
        # Merge all data sources
        merged_data = pd.merge(df, acs_data, on='GEOID', how='left')
        merged_data = pd.merge(merged_data, svi_data, on='GEOID', how='left')
        
        # Create features
        feature_cols = [
            'Rate_2022',
            'B19013_001E',  # Median household income
            'B15003_022E',  # Education metrics
            'B15003_023E',
            'B15003_024E',
            'B15003_025E'
        ]
        
        X = merged_data[feature_cols].values
        y = merged_data['label'].values
        
        return X, y 