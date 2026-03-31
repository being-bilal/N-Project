"""
This script rasterizes the volcano locations and computes the distance to the nearest volcano for each pixel.
The resulting raster is saved as "distance_to_volcano.tif" with ocean pixels masked out
"""

import rasterio 
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import os

with rasterio.open("/Users/mohammadbilal/Documents/Projects/N-Project/data/raw/Seismic_data.tif") as src:
    sample_crs = src.crs
    sample_shape = (src.height, src.width)
    sample_transform = src.transform
    pixel_size_deg = src.res[0] #pixel width in degrees 
    sample_meta = src.meta.copy()

pixel_size_km = pixel_size_deg * 111

world = gpd.read_file("/Users/mohammadbilal/Documents/Projects/N-Project/data/raw/borders_data/ne_10m_admin_0_countries.shp")
world = world.to_crs(sample_crs)

land_mask = rasterize(
    [(geom, 1) for geom in world.geometry],
    out_shape=sample_shape,
    transform=sample_transform,
    fill=0,
    dtype="uint8"
)
# land_mask: 1 = land, 0 = ocean

# --- Rasterize volcanoes ---
volcanoes = gpd.read_file("/Users/mohammadbilal/Documents/Projects/N-Project/data/processed/volcanoes.geojson")
volcanoes = volcanoes.to_crs(sample_crs)

volcano_grid = rasterize(
    [(geom, 1) for geom in volcanoes.geometry],
    out_shape=sample_shape,
    transform=sample_transform,
    fill=0,
    dtype="uint8"
)

# --- Distance transform ---
distance = distance_transform_edt(volcano_grid == 0) * pixel_size_km

distance[land_mask == 0] = -9999

# --- Save ---
os.makedirs("data/processed/rasters", exist_ok=True)
meta = sample_meta.copy()
meta.update(dtype="float32", count=1, nodata=-9999)

with rasterio.open("/Users/mohammadbilal/Documents/Projects/N-Project/data/processed/distance_to_volcano.tif", "w", **meta) as dst:
    dst.write(distance.astype("float32"), 1)

print("Done — distance to volcano saved (ocean masked)")