import numpy as np
import rasterio
import lightgbm as lgb  
import os
import pickle

with open('Models/LightGBM.pkl', 'rb') as f:
    trained_model = pickle.load(f)

feature_order = [
    'seismic_pga', 'dist_to_water', 'dist_to_fault', 'population', 
    'is_protected', 'dist_to_volcano', 'border_distance'
]

# Mapping feature names to their corresponding raster files
raster_paths = {
    'seismic_pga':     'maps/TestMaps/asia_seismic_pga.tif',
    'dist_to_water':   'maps/TestMaps/asia_dist_to_water.tif',
    'dist_to_fault':   'maps/TestMaps/asia_dist_to_fault.tif',
    'population':      'maps/TestMaps/asia_population.tif',
    'is_protected':    'maps/TestMaps/asia_is_protected.tif',
    'dist_to_volcano': 'maps/TestMaps/asia_dist_to_volcano.tif',
    'border_distance': 'maps/TestMaps/asia_border_distance.tif'
}

# Read the first raster to get the spatial dimensions and metadata template
with rasterio.open(raster_paths['seismic_pga']) as ref_raster:
    meta = ref_raster.meta.copy() 
    height = ref_raster.height
    width = ref_raster.width
    shape = (height, width)

total_pixels = height * width
print("Total pixels : ", total_pixels)

# array to hold the feature values for all pixels
feature_matrix = np.zeros((total_pixels, len(feature_order)), dtype=np.float32)
for idx, feature_name in enumerate(feature_order):
    with rasterio.open(raster_paths[feature_name]) as src:
        feature_matrix[:, idx] = src.read(1).flatten()

nodata_val = -9999.0
# Identify valid land cells (cells where NO feature equals -9999 and NO feature is NaN)
valid_land_mask = ~np.any(feature_matrix == nodata_val, axis=1) & ~np.any(np.isnan(feature_matrix), axis=1)
print(f"Total active terrestrial pixels to compute: {np.sum(valid_land_mask):,}")

# Initialize a completely empty master map filled entirely with -9999 background 
suitability_array_flat = np.full(total_pixels, nodata_val, dtype=np.float32)

if np.sum(valid_land_mask) > 0:
    print("\nBroadcasting machine learning model predictions over land cells...")
    X_inference = feature_matrix[valid_land_mask]
    predictions = trained_model.predict(X_inference)
    
    # Normalize predictions to 0.0–1.0 range for better interpretability
    current_min = predictions.min()
    current_max = predictions.max()
    print(f" -> Detected Raw Model Score Range: {current_min:.4f} to {current_max:.4f}")
    print(" -> Rescaling terrestrial grid values to a standard 0.0 to 1.0 index...")
    
    if current_max > current_min:
        predictions_normalized = (predictions - current_min) / (current_max - current_min)
    else:
        predictions_normalized = predictions  
        
    suitability_array_flat[valid_land_mask] = predictions_normalized

suitability_map_2d = suitability_array_flat.reshape(shape)
output_filename = "Predicted_raster.tif"
print(f"\nWriting spatial matrix back to georeferenced GeoTIFF format...")

meta.update(
    dtype=rasterio.float32,
    count=1,          
    nodata=nodata_val 
)

with rasterio.open(output_filename, "w", **meta) as dst:
    dst.write(suitability_map_2d, 1)

