import rasterio 
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import os

# Dataset used as the base for rasterisation
with rasterio.open("/Users/mohammadbilal/Documents/Projects/N-Project/data/raw/Seismic_data.tif") as src:
    sample_crs = src.crs
    sample_shape = src.shape
    sample_shape = (src.height, src.width)
    sample_transform = src.transform
    pixel_size_deg = src.res[0]  
    sample_meta = src.meta.copy()
    w, h = src.width, src.height
    l, b, r, t = src.bounds

    pixel_size_deg = src.res[0]  # degrees per pixel

# Approximate km per pixel (at mid-latitudes 1 degree ≈ 111km)
pixel_size_km = pixel_size_deg * 111

volcanoes = gpd.read_file("/Users/mohammadbilal/Documents/Projects/N-Project/data/processed/volcanoes.geojson")
volcanoes = volcanoes.to_crs(sample_crs)

# Rasterize — 1 where a volcano exists, 0 everywhere else
volcano_grid = rasterize(
    [(geom, 1) for geom in volcanoes.geometry],
    out_shape=sample_shape,
    transform=sample_transform,
    fill=0,
    dtype="uint8"
)

# Distance transform — gives distance in pixels, convert to km
distance = distance_transform_edt(volcano_grid == 0) * pixel_size_km

# Save
meta = sample_meta.copy()
meta.update(dtype="float32", count=1, nodata=-9999)
os.makedirs("data/raw/volcanoes", exist_ok=True)
with rasterio.open("data/raw/volcano.tif", "w", **meta) as dst:
    dst.write(distance.astype("float32"), 1)

print("Done — distance to volcano saved")