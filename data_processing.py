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
        """Fetch and process Social Vulnerability Index data."""
        # Load SVI data
        svi_path = 'SVI_2022_US.csv'
        if not os.path.exists(svi_path):
            raise FileNotFoundError(f"SVI data file not found: {svi_path}")

        # Read SVI data
        svi_data = pd.read_csv(svi_path)

        # Filter for Virginia
        va_svi = svi_data[svi_data['ST'] == '51']  # 51 is Virginia's FIPS code

        # Select relevant columns (using the correct column names from the SVI data)
        svi_columns = [
            'FIPS',  # Census tract FIPS code
            'RPL_THEMES',  # Overall SVI ranking
            'EP_POV150',  # Poverty (150% of poverty threshold)
            'EP_UNEMP',  # Unemployment
            'EP_HBURD',  # Housing burden (replacing EP_PCI)
            'EP_NOHSDP',  # No high school diploma
            'EP_AGE65',  # Age 65 and older
            'EP_AGE17',  # Age 17 and younger
            'EP_DISABL',  # Disability
            'EP_SNGPNT',  # Single-parent households
            'EP_MINRTY',  # Minority
            'EP_LIMENG',  # Limited English
            'EP_MUNIT',  # Multi-unit structures
            'EP_MOBILE',  # Mobile homes
            'EP_CROWD',  # Crowding
            'EP_NOVEH',  # No vehicle
            'EP_GROUPQ'  # Group quarters
        ]

        # Create GEOID from FIPS
        va_svi['GEOID'] = va_svi['FIPS'].astype(str).str.zfill(11)

        # Select and return relevant columns
        return va_svi[['GEOID'] + svi_columns[1:]]

    def create_spatial_features(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Create spatial features using tract geometries from TIGER/Line shapefiles."""
        # Path to the downloaded TIGER/Line shapefile
        shapefile_path = 'tl_2021_51_tract/tl_2021_51_tract.shp'
        if not os.path.exists(shapefile_path):
            raise FileNotFoundError(
                f"TIGER/Line shapefile not found: {shapefile_path}. "
                "Please ensure the shapefile is in the tl_2021_51_tract directory."
            )

        # Read the shapefile
        va_tracts = gpd.read_file(shapefile_path)

        # Project to UTM Zone 18N (appropriate for Virginia)
        va_tracts = va_tracts.to_crs('EPSG:32618')

        # Ensure GEOID is in the same format as our data
        va_tracts['GEOID'] = va_tracts['GEOID'].astype(str)

        # Convert df to GeoDataFrame by merging with va_tracts
        spatial_df = gpd.GeoDataFrame(
            pd.merge(df, va_tracts[['GEOID', 'geometry']], on='GEOID', how='left'),
            geometry='geometry',
            crs=va_tracts.crs
        )

        # Calculate spatial features
        spatial_df['centroid'] = spatial_df.geometry.centroid
        spatial_df['area'] = spatial_df.geometry.area  # in square meters
        spatial_df['perimeter'] = spatial_df.geometry.length  # in meters
        spatial_df['compactness'] = 4 * np.pi * spatial_df['area'] / (spatial_df['perimeter'] ** 2)

        # Calculate distance to state centroid (Richmond)
        richmond = Point([-77.4360, 37.5407])
        richmond_gdf = gpd.GeoDataFrame(geometry=[richmond], crs='EPSG:4326')
        richmond_gdf = richmond_gdf.to_crs('EPSG:32618')
        richmond_point = richmond_gdf.geometry.iloc[0]
        spatial_df['dist_to_richmond'] = spatial_df['centroid'].distance(richmond_point)  # in meters

        return spatial_df

    def prepare_features(self, df: pd.DataFrame, acs_data: pd.DataFrame, svi_data: pd.DataFrame) -> Tuple[
        np.ndarray, np.ndarray]:
        """Prepare final feature matrix and labels."""
        # Merge all data sources
        merged_data = pd.merge(df, acs_data, on='GEOID', how='left')
        merged_data = pd.merge(merged_data, svi_data, on='GEOID', how='left')

        # Create features
        feature_cols = [
            'Rate',  # Current suicide rate
            'B19013_001E',  # Median household income
            'B15003_022E',  # Education metrics
            'B15003_023E',
            'B15003_024E',
            'B15003_025E',
            # SVI features
            'RPL_THEMES',  # Overall SVI ranking
            'EP_POV150',  # Poverty (150% of poverty threshold)
            'EP_UNEMP',  # Unemployment
            'EP_HBURD',  # Housing burden
            'EP_NOHSDP',  # No high school diploma
            'EP_AGE65',  # Age 65 and older
            'EP_AGE17',  # Age 17 and younger
            'EP_DISABL',  # Disability
            'EP_SNGPNT',  # Single-parent households
            'EP_MINRTY',  # Minority
            'EP_LIMENG',  # Limited English
            'EP_MUNIT',  # Multi-unit structures
            'EP_MOBILE',  # Mobile homes
            'EP_CROWD',  # Crowding
            'EP_NOVEH',  # No vehicle
            'EP_GROUPQ'  # Group quarters
        ]

        # Convert all feature columns to numeric, coercing errors to NaN
        for col in feature_cols:
            if col in merged_data.columns:
                merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')
            else:
                print(f"Warning: Column {col} not found in data")

        # Select only numeric columns for feature matrix
        numeric_cols = merged_data[feature_cols].select_dtypes(include=[np.number]).columns
        print("Numeric columns available for features:", numeric_cols.tolist())

        # Handle missing values only for numeric columns
        merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].mean())

        # Create binary label: top 10% rate
        threshold = np.percentile(merged_data['Rate'], 90)
        merged_data['label'] = (merged_data['Rate'] >= threshold).astype(int)

        X = merged_data[numeric_cols].values
        y = merged_data['label'].values

        return X, y
