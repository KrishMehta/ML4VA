import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import numpy as np
from typing import Dict, List, Tuple


class Visualizer:
    def __init__(self):
        self.app = Dash(__name__)

    def create_risk_map(self,
                        gdf: gpd.GeoDataFrame,
                        risk_scores: np.ndarray,
                        tract_ids: List[str]) -> folium.Map:
        """Create an interactive risk map using Folium."""
        # Create base map centered on Virginia
        m = folium.Map(location=[37.5215, -78.6569], zoom_start=7)

        # Add risk scores to GeoDataFrame
        gdf['risk_score'] = pd.Series(risk_scores, index=tract_ids)

        # Create choropleth
        folium.Choropleth(
            geo_data=gdf.__geo_interface__,
            name='choropleth',
            data=gdf,
            columns=['GEOID', 'risk_score'],
            key_on='feature.properties.GEOID',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Risk Score'
        ).add_to(m)

        return m

    def create_dashboard(self,
                         gdf: gpd.GeoDataFrame,
                         model_results: Dict[str, Dict[str, float]],
                         risk_scores: Dict[str, np.ndarray]):
        """Create an interactive dashboard using Dash."""
        self.app.layout = html.Div([
            html.H1('Virginia Suicide Risk Mapping Dashboard'),

            # Model Performance Metrics
            html.Div([
                html.H2('Model Performance'),
                dcc.Graph(
                    figure=self._create_metrics_plot(model_results)
                )
            ]),

            # Risk Map
            html.Div([
                html.H2('Risk Map'),
                dcc.Graph(
                    figure=self._create_risk_map_plot(gdf, risk_scores)
                )
            ]),

            # Feature Importance
            html.Div([
                html.H2('Feature Importance'),
                dcc.Graph(
                    figure=self._create_feature_importance_plot()
                )
            ])
        ])

    def _create_metrics_plot(self, model_results: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create a bar plot of model performance metrics."""
        metrics = ['auroc', 'precision', 'recall']
        models = list(model_results.keys())

        fig = go.Figure()
        for metric in metrics:
            values = [model_results[model][metric] for model in models]
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=values
            ))

        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            yaxis_title='Score',
            xaxis_title='Model'
        )

        return fig

    def _create_risk_map_plot(self,
                              gdf: gpd.GeoDataFrame,
                              risk_scores: Dict[str, np.ndarray]) -> go.Figure:
        """Create an interactive risk map using Plotly."""
        fig = px.choropleth(
            gdf,
            geojson=gdf.geometry,
            locations=gdf.index,
            color='risk_score',
            color_continuous_scale='YlOrRd',
            title='Suicide Risk by Census Tract'
        )

        fig.update_geos(
            fitbounds="locations",
            visible=False
        )

        return fig

    def _create_feature_importance_plot(self) -> go.Figure:
        """Create a bar plot of feature importance."""
        # This would be implemented based on your model's feature importance
        # For now, returning a placeholder
        return go.Figure()

    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the Dash server."""
        self.app.run_server(debug=debug, port=port)
