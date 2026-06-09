"""
Rasterise vector data (rivers, borders) to create proximity masks.
For rivers, we create a categorical proximity mask:
0 = on water (river or ocean)
1 = within 5km 
2 = within 10km
3 = within 50km
4 = within 100km
5 = beyond 100km (too far — poor cooling water access)
"""

import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import os

BASE = "/Users/mohammadbilal/Documents/Projects/N-Project"

# Template raster 
with rasterio.open(f"{BASE}/data/raw/seismic_data.tif") as src:
    sample_crs       = src.crs
    sample_shape     = (src.height, src.width)
    sample_transform = src.transform
    sample_meta      = src.meta.copy()
    pixel_size_deg   = src.res[0]

pixel_size_km = pixel_size_deg * 111

# --- Land mask (1 = land, 0 = ocean) ---
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

# RIVERS — proximity mask with ocean as water source
rivers = gpd.read_file(f"{BASE}/data/raw/rivers_data/ne_10m_rivers_lake_centerlines.shp")
rivers = rivers.to_crs(sample_crs)
river_grid = rasterize(
    [(geom, 1) for geom in rivers.geometry],
    out_shape=sample_shape,
    transform=sample_transform,
    fill=0,
    dtype="uint8"
)

water_grid = river_grid.copy()
water_grid[land_mask == 0] = 1  # ocean = water present
# Distance transform from any water pixel (river OR ocean)
distance_km = distance_transform_edt(water_grid == 0) * pixel_size_km


proximity = np.zeros(sample_shape, dtype="uint8")
proximity[water_grid == 1]                                    = 0  # on water
proximity[(distance_km > 0)   & (distance_km <= 5)]          = 1  # 0–5 km
proximity[(distance_km > 5)   & (distance_km <= 10)]         = 2  # 5–10 km
proximity[(distance_km > 10)  & (distance_km <= 50)]         = 3  # 10–50 km
proximity[(distance_km > 50)  & (distance_km <= 100)]        = 4  # 50–100 km
proximity[distance_km > 100]                                  = 5  # beyond 100 km

# Pixels that are ocean themselves get category 0
proximity[land_mask == 0] = 0


meta = sample_meta.copy()
meta.update(dtype="uint8", count=1, nodata=255)

with rasterio.open(f"{BASE}/data/processed/rasters/water_proximity.tif", "w", **meta) as dst:
    dst.write(proximity, 1)


meta_float = sample_meta.copy()
meta_float.update(dtype="float32", count=1, nodata=-9999)
distance_continuous = distance_km.copy()
distance_continuous[land_mask == 0] = 0  # ocean = 0km distance

with rasterio.open(f"{BASE}/data/processed/rasters/distance_to_water.tif", "w", **meta_float) as dst:
    dst.write(distance_continuous.astype("float32"), 1)

