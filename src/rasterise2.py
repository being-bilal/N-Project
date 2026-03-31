"""
This script rasterizes country borders and computes the distance from each pixel to the nearest border
"""

import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import os

BASE = "/Users/mohammadbilal/Documents/Projects/N-Project"

with rasterio.open(f"{BASE}/data/raw/seismic_data.tif") as src:
    sample_crs       = src.crs
    sample_shape     = (src.height, src.width)
    sample_transform = src.transform
    sample_meta      = src.meta.copy()
    pixel_size_deg   = src.res[0]

pixel_size_km = pixel_size_deg * 111

# --- Land mask (to remove ocean pixels) ---
world = gpd.read_file("/Users/mohammadbilal/Documents/Projects/N-Project/data/raw/borders_data/ne_10m_admin_0_countries.shp")
world = world.to_crs(sample_crs)

land_mask = rasterize(
    [(geom, 1) for geom in world.geometry],
    out_shape=sample_shape,
    transform=sample_transform,
    fill=0,
    dtype="uint8"
)
os.makedirs(f"{BASE}/data/processed/rasters", exist_ok=True)

countries = gpd.read_file(f"{BASE}/data/raw/borders_data/ne_10m_admin_0_countries.shp")
countries = countries.to_crs(sample_crs)

# Extract boundary lines from country polygons
from shapely.ops import unary_union
border_lines = unary_union(countries.geometry.boundary)
borders_gdf  = gpd.GeoDataFrame(geometry=[border_lines], crs=sample_crs)

border_grid = rasterize(
    [(geom, 1) for geom in borders_gdf.geometry],
    out_shape=sample_shape,
    transform=sample_transform,
    fill=0,
    dtype="uint8"
)

distance_border = distance_transform_edt(border_grid == 0) * pixel_size_km
distance_border[land_mask == 0] = -9999  # mask ocean

with rasterio.open(f"{BASE}/data/processed/rasters/distance_to_border.tif", "w", **meta) as dst:
    dst.write(distance_border.astype("float32"), 1)

