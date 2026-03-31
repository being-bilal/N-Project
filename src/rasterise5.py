import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
import pandas as pd
import os

BASE = "/Users/mohammadbilal/Documents/Projects/N-Project"

# --- Template ---
with rasterio.open(f"{BASE}/data/raw/seismic_data.tif") as src:
    sample_crs       = src.crs
    sample_shape     = (src.height, src.width)
    sample_transform = src.transform
    sample_meta      = src.meta.copy()

# --- Land mask ---

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

# =====================================================================
# PROTECTED AREAS — binary exclusion mask
# 0 = inside protected area (excluded)
# 1 = outside protected area (acceptable)
# =====================================================================
print("Processing protected areas...")

# Load polygons
protected_poly = gpd.read_file(
    f"{BASE}/data/raw/protected_areas_data/WDPA_Apr2026_Public_shp-polygons.shp"
)
protected_poly = protected_poly.to_crs(sample_crs)
print(f"Loaded {len(protected_poly)} polygons")

# Load points — buffer to small polygons
protected_pts = gpd.read_file(
    f"{BASE}/data/raw/protected_areas_data/WDPA_Apr2026_Public_shp-points.shp"
)
protected_pts = protected_pts.to_crs("EPSG:3857")
protected_pts["geometry"] = protected_pts.geometry.buffer(500)
protected_pts = protected_pts.to_crs(sample_crs)
print(f"Loaded {len(protected_pts)} points (buffered to 500m)")

# Combine both layers
protected_all = gpd.GeoDataFrame(
    pd.concat([protected_poly, protected_pts], ignore_index=True),
    crs=sample_crs
)

# Rasterize — 1 where protected area exists
protected_grid = rasterize(
    [(geom, 1) for geom in protected_all.geometry if geom is not None],
    out_shape=sample_shape,
    transform=sample_transform,
    fill=0,
    dtype="uint8"
)

# Flip to exclusion logic
# 0 = protected (excluded)
# 1 = not protected (acceptable)
exclusion_mask = np.where(protected_grid == 1, 0, 1).astype("uint8")

# Ocean pixels get nodata
exclusion_mask[land_mask == 0] = 255

# --- Save ---
meta = sample_meta.copy()
meta.update(dtype="uint8", count=1, nodata=255)

with rasterio.open(f"{BASE}/data/processed/rasters/protected_areas_mask.tif", "w", **meta) as dst:
    dst.write(exclusion_mask, 1)

# --- Verification ---
print("\n--- Protected Areas Exclusion Mask ---")
land_pixels = exclusion_mask[land_mask == 1]
excluded  = (land_pixels == 0).sum()
safe      = (land_pixels == 1).sum()
total     = excluded + safe
print(f"  Excluded (protected):     {excluded:>10,} px  ({excluded/total*100:.1f}%)")
print(f"  Acceptable (unprotected): {safe:>10,} px  ({safe/total*100:.1f}%)")
print(f"\nSaved: protected_areas_mask.tif (binary 0/1)")