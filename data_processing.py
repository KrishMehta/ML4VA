import pandas as pd
import numpy as np
from census import Census
import os
from typing import Tuple, Dict, List
import geopandas as gpd
from shapely.geometry import Point
import requests
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging


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
        logging.info(f"[load_cdc_data] Loaded data shape: {df.shape}")
        logging.info(f"[load_cdc_data] Available columns: {list(df.columns)}")
        
        va = df[(df.ST_NAME == 'Virginia') & (df.Intent == 'All_Suicide')]
        logging.info(f"[load_cdc_data] Virginia suicide data shape: {va.shape}")
        
        # Fix column names (remove leading/trailing spaces)
        va.columns = va.columns.str.strip()
        
        # Check if we have the expected columns
        if 'Rate' in va.columns:
            # Replace -999 values with NaN
            va['Rate'] = va['Rate'].replace(-999.0, np.nan)
            va['Rate'] = va['Rate'].replace(-999, np.nan)
            
            # Log statistics after cleaning
            logging.info(f"[load_cdc_data] Rate statistics after cleaning - "
                        f"Min: {va['Rate'].min():.2f}, Max: {va['Rate'].max():.2f}, "
                        f"Mean: {va['Rate'].mean():.2f}, Non-null: {va['Rate'].notna().sum()}")
        else:
            logging.warning("[load_cdc_data] 'Rate' column not found in data!")
            
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
        va_svi = svi_data[svi_data['ST'] == 51]  # 51 is Virginia's FIPS code (as integer)

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
        
        # If df is provided, filter to only those GEOIDs
        if 'GEOID' in df.columns:
            # Get unique GEOIDs from the input data
            data_geoids = df['GEOID'].unique()
            logging.info(f"[create_spatial_features] Filtering shapefile to {len(data_geoids)} GEOIDs from input data")
            va_tracts = va_tracts[va_tracts['GEOID'].isin(data_geoids)]
            logging.info(f"[create_spatial_features] Filtered shapefile has {len(va_tracts)} tracts")

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
        np.ndarray, np.ndarray, np.ndarray]:
        """Prepare final feature matrix and labels."""
        logging.info(f"[prepare_features] Input data shapes - CDC: {df.shape}, ACS: {acs_data.shape}, SVI: {svi_data.shape}")
        
        # Filter for 2022 and 2023 data
        df_2022 = df[df['Period'] == '2022'].copy()
        df_2023 = df[df['Period'] == '2023'].copy()
        
        # Extract individual census tracts from the NAME column
        # CDC data aggregates multiple tracts, we need to disaggregate them
        def expand_cdc_data(df_year):
            rows = []
            for _, row in df_year.iterrows():
                if pd.notna(row['NAME']):
                    # Split the comma-separated tract FIPS codes
                    tract_fips_list = [fips.strip() for fips in row['NAME'].split(',')]
                    # Create a row for each individual tract
                    for tract_fips in tract_fips_list:
                        new_row = row.copy()
                        new_row['GEOID'] = tract_fips  # Use the actual tract FIPS as GEOID
                        rows.append(new_row)
            return pd.DataFrame(rows)
        
        # Expand the data to individual tracts
        df_2022_expanded = expand_cdc_data(df_2022)
        df_2023_expanded = expand_cdc_data(df_2023)
        
        logging.info(f"[prepare_features] Expanded 2022 data: {len(df_2022)} -> {len(df_2022_expanded)} rows")
        logging.info(f"[prepare_features] Expanded 2023 data: {len(df_2023)} -> {len(df_2023_expanded)} rows")
        
        # Merge 2022 and 2023 data
        merged_years = pd.merge(
            df_2022_expanded[['GEOID', 'Rate']].rename(columns={'Rate': 'Rate_2022'}),
            df_2023_expanded[['GEOID', 'Rate']].rename(columns={'Rate': 'Rate_2023'}),
            on='GEOID',
            how='inner'
        )
        
        # Remove rows with missing rates
        merged_years = merged_years.dropna(subset=['Rate_2022', 'Rate_2023'])
        logging.info(f"[prepare_features] Merged years data shape: {merged_years.shape}")
        
        # Merge all data sources
        merged_data = pd.merge(merged_years, acs_data, on='GEOID', how='left')
        logging.info(f"[prepare_features] After ACS merge: {merged_data.shape}")
        
        merged_data = pd.merge(merged_data, svi_data, on='GEOID', how='left')
        logging.info(f"[prepare_features] After SVI merge: {merged_data.shape}")
        
        # Create features - include 2022 rate as a feature
        feature_cols = [
            'Rate_2022',  # Previous year's rate as predictor
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
                logging.warning(f"[prepare_features] Column {col} not found in data")

        # Select only numeric columns for feature matrix
        numeric_cols = merged_data[feature_cols].select_dtypes(include=[np.number]).columns
        logging.info(f"[prepare_features] Numeric columns available for features: {numeric_cols.tolist()}")

        # Create feature matrix
        X = merged_data[numeric_cols].values

        # Handle missing values using SimpleImputer with a different strategy
        imputer = SimpleImputer(strategy='constant', fill_value=0)  # Fill missing values with 0
        X = imputer.fit_transform(X)
        
        # Scale features to have zero mean and unit variance
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Create binary label: top 10% rate
        threshold = np.percentile(merged_data['Rate_2023'], 90)
        y = (merged_data['Rate_2023'] >= threshold).astype(int).values

        # More detailed debugging
        logging.info(f"[prepare_features] Feature matrix shape: {X.shape}")
        logging.info(f"[prepare_features] Label vector shape: {y.shape}")
        logging.info(f"[prepare_features] Label distribution - Class 0: {np.sum(y==0)}, Class 1: {np.sum(y==1)}")
        logging.info(f"[prepare_features] Feature statistics - Mean: {np.mean(X):.3f}, Std: {np.std(X):.3f}, % zeros: {np.mean(X==0)*100:.1f}%")
        logging.info(f"[prepare_features] Number of features with all zeros: {np.sum(np.all(X==0, axis=0))}")

        return X, y, merged_data['GEOID'].values
