"""
this script rasterizes fault lines and computes a binary exclusion mask based on distance to faults
0 = within 10km of fault (excluded)
1 = beyond 10km (acceptable)
"""
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import os

BASE = "/Users/mohammadbilal/Documents/Projects/N-Project"

# --- Template ---
with rasterio.open(f"{BASE}/data/raw/seismic_data.tif") as src:
    sample_crs       = src.crs
    sample_shape     = (src.height, src.width)
    sample_transform = src.transform
    sample_meta      = src.meta.copy()
    pixel_size_deg   = src.res[0]

pixel_size_km = pixel_size_deg * 111

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
# FAULT LINES — binary exclusion mask
# 0 = within 100km of fault (excluded)
# 1 = beyond 100km (acceptable)
# =====================================================================
print("Processing fault lines...")

faults = gpd.read_file(
    f"{BASE}/data/raw/faults_data/faults_data/gem_active_faults.shp"
)
faults = faults.set_crs("EPSG:4326")
faults = faults.to_crs(sample_crs)
print(f"Loaded {len(faults)} fault features")

# Rasterize fault lines
fault_grid = rasterize(
    [(geom, 1) for geom in faults.geometry],
    out_shape=sample_shape,
    transform=sample_transform,
    fill=0,
    dtype="uint8"
)

# Compute distance from every pixel to nearest fault
distance_fault_km = distance_transform_edt(fault_grid == 0) * pixel_size_km

# Apply 10km exclusion threshold
# 0 = too close to fault (bad)
# 1 = safe distance from fault (good)
fault_mask = np.where(distance_fault_km >= 10, 1, 0).astype("uint8")

# Ocean pixels get nodata
fault_mask[land_mask == 0] = 255

# --- Save binary mask ---
meta = sample_meta.copy()
meta.update(dtype="uint8", count=1, nodata=255)

with rasterio.open(f"{BASE}/data/processed/rasters/fault_exclusion_mask.tif", "w", **meta) as dst:
    dst.write(fault_mask, 1)

# --- Also save continuous distance for reference ---
meta_float = sample_meta.copy()
meta_float.update(dtype="float32", count=1, nodata=-9999)
distance_fault_km[land_mask == 0] = -9999

with rasterio.open(f"{BASE}/data/processed/rasters/distance_to_fault.tif", "w", **meta_float) as dst:
    dst.write(distance_fault_km.astype("float32"), 1)

# --- Verification ---
print("\n--- Fault Exclusion Mask ---")
land_pixels = fault_mask[land_mask == 1]
excluded  = (land_pixels == 0).sum()
safe      = (land_pixels == 1).sum()
total     = excluded + safe
print(f"  Excluded (within 100km of fault): {excluded:>10,} px  ({excluded/total*100:.1f}%)")
print(f"  Safe     (beyond 100km of fault): {safe:>10,} px  ({safe/total*100:.1f}%)")
print(f"\nSaved: fault_exclusion_mask.tif  (binary 0/1)")
print(f"Saved: distance_to_fault.tif     (continuous km, for reference)")