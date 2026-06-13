## Models Used for the Site selection of the Nuclear Power Plant

In this project, multiple tree based regression models are trained on the dataset containing 1,913 coordinates optimized for training machine learning models on nuclear power plant siting suitability. It introduces a continuous target spectrum to map, blending 411 operational plants (Target=1.0), 500 marginal decoy sites with hidden environmental risks (Target=0.5), and 1,002 unsuitable points (Target=0.0). Along with machine learning moels, the process of site suitability is also carried out using analytical method of WLC (weighted linear combination) using analytic Hierarchy Process to determine the weights of each factor.

---
### XGBoosting
XGBoost (Extreme Gradient Boosting) is an advanced and highly optimized implementation of Gradient Boosting that is widely used for structured/tabular data problems such as classification, regression, ranking, and prediction tasks.
Like standard Gradient Boosting, XGBoost builds decision trees sequentially:

* Start with an initial prediction.
* Compute the residue (gradients) of the current model.
* Train a new tree to predict those errors.
* Add the new tree to the model.
* Repeat until the loss is minimized.
* But unlike the traditional gradient boost model it uses L2 regularisation and tree proning using a gain funtion that is calculated using the gradient and hessian of the instances of the dataset.

---

### LightGBM 
It is an implementation of the gradient boosting model just like XGBoost that is designed to provide high accuracy on large dataset while comsuning less memory and time, Its main innovations are histogram-based splitting, which reduces the number of split candidates by grouping continuous feature values into bins, and leaf-wise tree growth, which expands the leaf that provides the greatest reduction in loss rather than growing all nodes at the same depth. This allows it to save memory and perform much better on large datasets compared to traditional Gradient Boosting model.

--- 
### Weighted Linear Combination 
Weighted linear combination is a traditional method in GIS suitability framework where a final suitability score is calculated by multiplying standardized values of the factors such as population, distance to water and pga in this case by their weights and summing the results. (Y = Wx). For this weights are to be assigned to each factor. this can be done arbitarily using the importance of each fator as a criteria but to eliminate purely arbitrary weight assignment within WLC, the Analytic Hierarchy Process (AHP) is used. AHP is a mathematically rigorous decision-making framework that derives objective weights through a series of structured, matrix-based pairwise comparisons.
It is executed in four distinct steps as follows: 
* Matrix Evaluation (AHP): A 7×7 (number of factors = 7) pairwise matrix was constructed to rank your specific columns, using this matrix the weights of the factors were determined.
* Scaling: Because raw units (distances and pga values) cannot be added together directly, normalized values between 0.0 (highly dangerous/unsuitable) and 1.0 (perfectly safe/ideal) are used.
* Then For every single coordinate, we multiply those normalized values by their exact AHP weights and added them up to output a final traditional suitability prediction.