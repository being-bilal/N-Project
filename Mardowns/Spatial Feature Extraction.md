
#### Factors Used Feature Extraction 
- Earthquake ( Peak Ground Acceleration) 

- The necessity for a reliable and sufficient supply of cooling water (Raster vector data (rivers, borders) to create proximity masks. For rivers, we create a categorical proximity mask: 0 = on water (river or ocean), 1 = within 5km, 2 = within 10km, 3 = within 50km, 4 = within 100km, 5 = beyond 100km (too far — poor cooling water access)

- the presence of an active geological fault within the immediate vicinity of a site (raster fault lines and computes a binary exclusion mask based on distance to faults: 0 = within 10km of fault (excluded), 1 = beyond 10km (acceptable))

- proximity of the site to population centers (population density at each pixel)

- Siting must avoid areas dedicated to specific uses, such as national parks (polygonal mask for all protected area)

- Frequency and distribution of volcanic eruptions (raster with volcano locations and computes the distance to the nearest volcano for each pixel.)

- Proximity to International Borders (country borders and computes the distance from each pixel to the nearest border)

--- 

#### Creating Negative Samples
Instead of picking random points across the globe for "unsuitable" class, we are choosing 3000 targeted points split into three groups:

1. **1,000 Obvious Fakes:** Places that are physically impossible 

2. **1,000 Tricky Fakes:** Places just outside the line 
    
3. **1,000 Decoy Fakes:** Places that look perfect (flat land near water) but are highly dangerous for some reason (on top of an earthquake fault line).

---

#### Handling Country Borders
We are adding a Region column to the table and using One-Hot Encoding to turn it into a format XGBoost can read. Nuclear plants in Europe are often built right on international borders due to shared rivers, while plants in Asia are kept far away from borders for national security. Without telling the model _where_ the point is located, XGBoost will get confused by these contradicting human rules and give flawed predictions.

---

#### How to Split Data (Test/Train)
we rejected standard random splitting. Instead, we are using a **Region-Based Split**. You will train your model on data from certain continents (e.g., Europe and North America) and test its accuracy by making it predict on a completely unseen continent.
Nearby geographic points share almost identical features (spatial autocorrelation). If you use a standard random split, the model will just memorize regional landscapes rather than learning universal engineering principles.

---

#### The Non-ML Comparison
we will build a separate, traditional engineering model called **Weighted Linear Combination (WLC)** using simple math equations and expert-assigned weights. we will compare its results against your XGBoost model. In real-world civil engineering, black-box AI is rarely trusted blindly. By creating a transparent, rule-based baseline map, you can rigorously prove exactly where XGBoost outperforms traditional engineering logic and where it uncovers hidden non-linear relationships.
