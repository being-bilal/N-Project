import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Read CSV
df = pd.read_csv("/Users/mohammadbilal/Documents/Projects/N-Project/Dataset/Dataset.csv")

# Create geometry from longitude and latitude columns
geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=geometry,
    crs="EPSG:4326"  # WGS84 coordinate system
)

# Save as GeoJSON
gdf.to_file("output.geojson", driver="GeoJSON")

print("GeoJSON file saved as output.geojson")