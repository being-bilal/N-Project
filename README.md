# Nuclear Power Plant Platform : 
 
A geospatial ML project that answers two questions:
- **Where should new nuclear plants be built?**
- **How dangerous are the ones we already have?**
 
---
 
## Goal : 
 
**Site Suitability** — A Random Forest model trained on existing plant locations scores every pixel in the study region from 0 (unsuitable) to 1 (highly suitable), outputting a GeoTIFF raster.
 
**Hazard Exposure** — Wind-adjusted IAEA emergency zones (PAZ 5km / UPZ 30km / LCPZ 300km) are generated per plant, with population counts and K-Means risk tiers (Low / Medium / High / Critical).
 
Both analyses combine into a single interactive browser map.
 
---