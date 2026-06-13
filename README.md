# Nuclear Power Plant Site Suitability Analysis

A geospatial machine learning project for predicting suitable sites for nuclear power plants using tree-based ensemble models and traditional GIS-based weighted linear combination.

--- 
![Alt Text](https://github.com/being-bilal/N-Project/blob/main/Images/FinalMap.png)
---

## Reference Standard

This project is developed in accordance with the International Atomic Energy Agency (IAEA) SSR-1: Safety of Nuclear Power Plants — Site Evaluation, which defines the safety requirements for selecting sites for nuclear installations.

---

## Factors Comsidered for Site Selection

Derived from IAEA SSR-1 requirements, the following siting criteria were considered:

- **Seismic Value** — Peak Ground Acceleration (PGA) at the site
- ** Water Supply** — Reliability and sufficiency of water sources for reactor cooling
- **Active Geological Faults** — Presence of active fault lines within the range of the Powerplant
- **Population Proximity** — Distance and density of nearby population centers
- **Protected Areas** — Avoidance of areas with dedicated land use, such as national parks
- **Volcanic Activity** — Frequency and spatial distribution of volcanic eruptions
- **International Borders** — Proximity to national borders
- **Terrain Elevation** — Elevation of the terrain at the proposed power plant site (Terrain elevation was not included in the final site selection due to unavailability of the appropriate publicly available dataset)


![Alt Text](https://github.com/being-bilal/N-Project/blob/main/Images/collage.png)

---

## Datasets Used

| # | Factor | Dataset | Source |
|---|--------|---------|--------|
| 1 | Seismic (PGA) | GSHAP | gfz-potsdam.de |
| 2 | Cooling Water | HydroRIVERS | hydrosheds.org |
| 3 | Active Faults | GEM Global Faults | github.com/GEMScienceTools |
| 4 | Population | WorldPop 1km | hub.worldpop.org |
| 5 | Protected Areas | WDPA | protectedplanet.net |
| 6 | Borders | Natural Earth | naturalearthdata.com |
| 7 | Volcanic Data | GVP Holocene | volcano.si.edu |

---

## Models Used for Site Suitability Prediction

In this project, multiple tree based regression models are trained on the dataset containing 1,913 coordinates optimized for training machine learning models on nuclear power plant siting suitability. It introduces a continuous target spectrum to map, blending 411 operational plants (Target=1.0), 500 marginal decoy sites with hidden environmental risks (Target=0.5), and 1,002 unsuitable points (Target=0.0). Along with machine learning moels, the process of site suitability is also carried out using analytical method of WLC (weighted linear combination) using analytic Hierarchy Process to determine the weights of each factor.

---

### XGBoost

XGBoost (Extreme Gradient Boosting) is an advanced and highly optimized implementation of Gradient Boosting that is widely used for structured/tabular data problems such as classification, regression, ranking, and prediction tasks.
Like standard Gradient Boosting, XGBoost builds decision trees sequentially:

* Start with an initial prediction.
* Compute the residue (gradients) of the current model.
* Train a new tree to predict those errors.
* Add the new tree to the model.
* Repeat until the loss is minimized.
* But unlike the traditional gradient boost model it uses L2 regularisation and tree proning using a gain funtion that is calculated using the gradient and hessian of the instances of the dataset.
* Unlike traditional methods that rely on human assumptions, XGBoost automatically calculates the importance of each spatial feature based on how much it improves the model's prediction accuracy during training.
![Alt Text](https://github.com/being-bilal/N-Project/blob/a9f44b28167af2bff22313a409dacbb4647d6d8d/Graphs/XGboost_Feature_Importance.png)
---

### LightGBM

It is an implementation of the gradient boosting model just like XGBoost that is designed to provide high accuracy on large dataset while comsuning less memory and time, Its main innovations are histogram-based splitting, which reduces the number of split candidates by grouping continuous feature values into bins, and leaf-wise tree growth, which expands the leaf that provides the greatest reduction in loss rather than growing all nodes at the same depth. This allows it to save memory and perform much better on large datasets compared to traditional Gradient Boosting model.

---

### Weighted Linear Combination (WLC)

Weighted linear combination is a traditional method in GIS suitability framework where a final suitability score is calculated by multiplying standardized values of the factors such as population, distance to water and pga in this case by their weights and summing the results. (Y = Wx). For this weights are to be assigned to each factor. this can be done arbitarily using the importance of each fator as a criteria but to eliminate purely arbitrary weight assignment within WLC, the Analytic Hierarchy Process (AHP) is used. AHP is a mathematically rigorous decision-making framework that derives objective weights through a series of structured, matrix-based pairwise comparisons.
It is executed in four distinct steps as follows: 

* Matrix Evaluation (AHP): A 7×7 (number of factors = 7) pairwise matrix was constructed to rank your specific columns, using this matrix the weights of the factors were determined. 
![Alt Text](https://github.com/being-bilal/N-Project/blob/a9f44b28167af2bff22313a409dacbb4647d6d8d/Graphs/AHP_Matrix_Heatmap.png)
* Scaling: Because raw units (distances and pga values) cannot be added together directly, normalized values between 0.0 (highly dangerous/unsuitable) and 1.0 (perfectly safe/ideal) are used.
* Then For every single coordinate, we multiply those normalized values by their exact AHP weights and added them up to output a final traditional suitability prediction.
![Alt Text](https://github.com/being-bilal/N-Project/blob/a9f44b28167af2bff22313a409dacbb4647d6d8d/Graphs/AHP_Weights_Bar.png)
---

## Feature Extraction

The following geospatial feature extraction procedures are applied to each factor:

- **Peak Ground Acceleration (PGA):** Raw GSHAP raster values extracted per coordinate.
- **Cooling Water (Rivers):** Raster-vector distance analysis producing a categorical proximity mask:
  - `0` = On water (river or ocean)
  - `1` = Within 5 km
  - `2` = Within 10 km
  - `3` = Within 50 km
  - `4` = Within 100 km
  - `5` = Beyond 100 km (poor cooling water access)
- **Active Faults:** Binary exclusion mask based on distance to fault lines:
  - `0` = Within 10 km of a fault (excluded)
  - `1` = Beyond 10 km (acceptable)
- **Population:** Population density extracted at each pixel from WorldPop 1 km raster.
- **Protected Areas:** Polygonal mask applied for all WDPA-registered protected areas.
- **Volcanism:** Distance from each coordinate to the nearest Holocene volcano.
- **International Borders:** Distance from each coordinate to the nearest country border.

---

## Dataset Construction

### Creating Negative Samples

Instead of selecting random global points as "unsuitable" examples, **3,000 targeted negative points** are generated across three categories to create a challenging, realistic training signal:

| Group | Count | Description |
|-------|-------|-------------|
| Obvious Fakes | 1,000 | Physically impossible locations |
| Tricky Fakes | 1,000 | Sites just outside acceptable thresholds — close to faults, borders, or volcanoes |
| Decoy Fakes | 1,000 | Sites that appear ideal (flat terrain, near water) but are disqualified by a single critical hazard |

![Alt Text](https://github.com/being-bilal/N-Project/blob/main/Images/datapoints-target.png)


### Handling Country Borders

A `Region` column is added to the dataset. This is necessary because nuclear siting norms differ significantly by geopolitical context:

- In **Europe**, plants are often built near international borders due to shared rivers used for cooling.
- In **Asia**, plants are kept far from borders for national security reasons.

Without encoding regional context, the model encounters contradicting spatial patterns and produces flawed predictions.

---

### Train/Test Split Strategy

Standard random splitting is rejected in favor of a **Region-Based Split**:

- The model is trained on data from select continents.
- It is tested on a completely unseen continent to evaluate generalization.

Nearby geographic coordinates share nearly identical features due to spatial autocorrelation. A random split would allow the model to memorize regional landscapes rather than learning transferable, universal engineering principles.

![Alt Text](https://github.com/being-bilal/N-Project/blob/main/Images/datapoint-regions.png)


---

### ML vs. Traditional Engineering Comparison

A separate **Weighted Linear Combination (WLC)** model is built as a transparent, rule-based engineering baseline using expert-assigned AHP weights. Its predictions are directly compared against XGBoost and LightGBM outputs to:

- Rigorously identify where ML models outperform traditional engineering logic.
- Reveal non-linear relationships and hidden interactions that WLC cannot capture.
- Provide interpretable validation that satisfies real-world civil engineering practice, where black-box AI models are rarely trusted without a comparable deterministic benchmark.

## Results
---
### Machine Learning vs. Traditional Methods

To evaluate the robustness of our framework, we tested the models on a completely unseen holdout region (Asia) to simulate real-world generalization. The performance of the machine learning models (LightGBM and XGBoost) was compared against the traditional Weighted Linear Combination (WLC) baseline.

**Holdout Region (Asia) Metrics:**

* **LightGBM:** $R^2$ = 0.7445 | MAE = 0.1282
* **XGBoost:** $R^2$ = 0.7314 | MAE = 0.1243
* **Traditional WLC:** $R^2$ = -0.0688 | MAE = 0.4371

![Alt Text](https://github.com/being-bilal/N-Project/blob/a9f44b28167af2bff22313a409dacbb4647d6d8d/Graphs/Actual_Performance_Comparison.png)

The traditional WLC method yielded a negative $R^2$ score. This massive performance gap exists because WLC fails to capture strict non-linear safety thresholds. In WLC, a site with a massive population but excellent water access and zero seismic risk might still average out to a "moderate" suitability score. However, tree-based models like LightGBM and XGBoost successfully learned strict engineering vetoes: *if population density exceeds a certain safety threshold, the suitability must drop to absolute zero, regardless of how good the other factors are.* The ML models proved highly capable of learning these complex, real-world IAEA safety constraints natively from the data without requiring hard-coded rules.

### Geographic Distribution of the Suitability Index
![Alt Text](https://github.com/being-bilal/N-Project/blob/main/Graphs/Distribution.png)
When projecting the model's predictions across the continuous geography of the test continent. The vast majority of the continent's landmass falls into the **0.0 to 0.2** (Highly Unsuitable) range. This is an expected and accurate geographic reality: massive portions of land are eliminated due to compounding factors such as high population density, mountainous terrain, protected ecological reserves, or extreme distance from reliable cooling water.

Only a microscopic fraction of the total land area achieves a score of **0.8 to 1.0** (Highly Suitable). This validates the model's accuracy, demonstrating that finding an optimal nuclear site that perfectly balances geohazard safety, ecological preservation, and hydrological access is an exceptionally rare geographic anomaly.
