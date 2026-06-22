# Ames Housing Price Prediction

## Project Overview

This project analyzes the Ames Housing dataset and builds an end-to-end machine learning pipeline to predict house sale prices.

The primary goal of this project was not to maximize model performance, but to practice and demonstrate a complete machine learning workflow on a large real-world dataset, including:

* Data Understanding
* Exploratory Data Analysis (EDA)
* Missing Value Handling
* Feature Encoding
* Preprocessing Pipeline Design
* Baseline Regression Modeling

The project emphasizes understanding the meaning of each feature and making preprocessing decisions based on data exploration rather than applying generic techniques.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Dataset

The dataset contains residential property information from Ames, Iowa and includes:

* 2,930 observations
* 80+ features
* Numerical and categorical variables
* Significant missing values
* Multiple ordinal categorical features
* House sale prices as the target variable

### Target Variable

**SalePrice** – Final sale price of the house.

---

## Project Workflow

### 1. Data Understanding

A detailed column-by-column analysis was performed to understand:

* Feature meaning
* Data types
* Missing value patterns
* Encoding requirements
* Potential feature importance

A preprocessing strategy was created before any modeling was performed.

---

### 2. Exploratory Data Analysis

EDA was performed to identify important predictors of house prices.

#### Numerical Analysis

Correlation analysis was conducted for all numerical features.

Top numerical predictors:

| Feature       | Correlation |
| ------------- | ----------- |
| Overall Qual  | 0.799       |
| Gr Liv Area   | 0.707       |
| Garage Cars   | 0.648       |
| Garage Area   | 0.640       |
| Total Bsmt SF | 0.632       |
| 1st Flr SF    | 0.622       |
| Year Built    | 0.558       |

#### Categorical Analysis

Median SalePrice comparisons were used to evaluate categorical features.

Strongest categorical predictors:

| Feature      | Price Difference |
| ------------ | ---------------- |
| Exter Qual   | 4.36x            |
| Bsmt Qual    | 3.76x            |
| Neighborhood | 3.60x            |
| Kitchen Qual | 3.02x            |

---

## Key Findings

### Overall Quality is the strongest predictor

* Correlation with SalePrice: **0.799**
* House prices increase consistently as quality rating increases.

### Living Area strongly influences price

* Correlation with SalePrice: **0.707**
* Larger homes generally sell for higher prices.

### Neighborhood matters significantly

Highest median price:

* StoneBr → $319,000

Lowest median price:

* MeadowV → $88,250

This represents a **3.6x difference** in median house prices.

### Quality-related features dominate pricing

Features such as:

* Overall Qual
* Exter Qual
* Kitchen Qual
* Bsmt Qual
* Garage Finish

show strong ordinal relationships with SalePrice.

---

## Missing Value Handling

Different strategies were used depending on the meaning of the missing values.

### Imputed with "None"

Examples:

* Pool QC
* Fireplace Qu
* Garage Qual
* Garage Cond
* Garage Finish
* Bsmt Qual
* Bsmt Cond
* BsmtFin Type 1
* BsmtFin Type 2

These missing values generally represented the absence of a feature.

### Imputed with Median

* Lot Frontage
* Garage Yr Blt
* Garage Area

### Imputed with Most Frequent

* Electrical
* Garage Cars

### Imputed with Zero

* Bsmt Full Bath
* Bsmt Half Bath
* BsmtFin SF1
* BsmtFin SF2
* Bsmt Unf SF
* Total Bsmt SF

---

## Feature Encoding

### Ordinal Encoding

Applied to naturally ordered categorical variables:

- Overall Qual (1 < 2 < ... < 10)
- Overall Cond (1 < 2 < ... < 10)

#### Quality-Based Features

- Exter Qual (Po < Fa < TA < Gd < Ex)
- Exter Cond (Po < Fa < TA < Gd < Ex)
- Kitchen Qual (Po < Fa < TA < Gd < Ex)
- Heating QC (Po < Fa < TA < Gd < Ex)

#### Quality Features with "None" Category

- Bsmt Qual (None < Po < Fa < TA < Gd < Ex)
- Bsmt Cond (None < Po < Fa < TA < Gd < Ex)
- Pool QC (None < Fa < TA < Gd < Ex)
- Fireplace Qu (None < Po < Fa < TA < Gd < Ex)
- Garage Qual (None < Po < Fa < TA < Gd < Ex)
- Garage Cond (None < Po < Fa < TA < Gd < Ex)

#### Basement Features

- Bsmt Exposure (No Basement < No < Mn < Av < Gd)
- BsmtFin Type 1 (None < Unf < LwQ < Rec < BLQ < ALQ < GLQ)
- BsmtFin Type 2 (None < Unf < LwQ < Rec < BLQ < ALQ < GLQ)

#### Other Ordered Features

- Garage Finish (None < Unf < RFn < Fin)
- Functional (Sal < Sev < Maj2 < Maj1 < Mod < Min2 < Min1 < Typ)
- Paved Drive (N < P < Y)

### One-Hot Encoding

Applied to nominal categorical variables with no natural ordering:

- Neighborhood
- MS Zoning
- House Style
- Condition 1
- Condition 2
- Bldg Type
- Street
- Lot Config
- Roof Style
- Exterior 1st
- Exterior 2nd
- Foundation
- Heating
- Sale Type
- Sale Condition
- Lot Shape
- Land Contour
- Land Slope
- Roof Matl
- Central Air
- Electrical

---

## Features Removed

The following features were dropped:

* PID
* Order
* Utilities
* Garage Type
* Mas Vnr Type
* Alley
* Fence
* Misc Feature

Reason:

* Identifier columns or features with little predictive value.

---

## Model Training & Evaluation

Three regression models were trained and evaluated using the same preprocessing pipeline.

### 1. Linear Regression

Linear Regression was used as a baseline model to evaluate the effectiveness of the preprocessing and feature engineering strategy.

| Metric | Score  |
| ------ | ------ |
| R²     | 0.856  |
| RMSE   | 34,011 |
| MAE    | 14,835 |

#### Observations

* Achieved strong predictive performance despite its simplicity.
* Demonstrated that the preprocessing strategy successfully captured most of the important information in the dataset.
* Provided an interpretable baseline for comparison with tree-based models.

---

### 2. Decision Tree Regressor

A Decision Tree Regressor was tuned using GridSearchCV.

#### Best Parameters

```python
{
    'max_depth': 10,
    'max_features': None,
    'min_samples_leaf': 5,
    'min_samples_split': 20
}
```

#### Cross Validation Score

```text
R² = 0.787
```

#### Test Performance

| Metric | Score  |
| ------ | ------ |
| R²     | 0.841  |
| RMSE   | 35,660 |
| MAE    | 22,682 |

#### Observations

* Performance was slightly lower than Linear Regression.
* Indicates that the dataset contains strong linear and additive relationships.
* Tree-based splits alone were insufficient to outperform the linear baseline.

---

### 3. Random Forest Regressor

A Random Forest Regressor was trained to capture nonlinear relationships and feature interactions.

| Metric | Score      |
| ------ | ---------- |
| R²     | **0.906**  |
| RMSE   | **27,452** |
| MAE    | 16,098     |

#### Observations

* Achieved the best overall performance.
* Improved R² from 0.856 to 0.906.
* Reduced RMSE by approximately $6,500 compared to Linear Regression.
* Demonstrated the ability to capture nonlinear relationships within the housing data.

---

## Model Comparison

| Model             | R²        | RMSE       | MAE    |
| ----------------- | --------- | ---------- | ------ |
| Linear Regression | 0.856     | 34,011     | 14,835 |
| Decision Tree     | 0.841     | 35,660     | 22,682 |
| Random Forest     | **0.906** | **27,452** | 16,098 |

### Best Performing Model

**Random Forest Regressor** achieved the strongest predictive performance with:

* R² = 0.906
* RMSE = $27,452
* MAE = $16,098

The model explains approximately **90.6% of the variance** in house prices while producing the lowest overall prediction error.

---

## Feature Importance (Random Forest)

Random Forest feature importance aligned closely with the findings from Exploratory Data Analysis.

| Feature       | Importance |
| ------------- | ---------- |
| Overall Qual  | 58.1%      |
| Gr Liv Area   | 8.9%       |
| Garage Area   | 4.0%       |
| Garage Cars   | 4.0%       |
| Total Bsmt SF | 3.2%       |

### Key Insight

Overall Quality was by far the most influential feature in predicting house prices, accounting for more than half of the model's total feature importance. Living area, garage characteristics, and basement size were also major contributors.

These findings strongly validate the conclusions drawn during EDA, where Overall Quality showed the strongest relationship with SalePrice.

---

### Final Conclusion

This project demonstrates a complete machine learning workflow, beginning with data understanding and exploratory analysis, followed by preprocessing, feature engineering, model development, and evaluation.

Among the models tested, Random Forest Regressor achieved the best performance with an R² score of **0.906**, confirming that house quality, living area, garage characteristics, and basement size are the most influential factors affecting residential property prices in Ames, Iowa.



## Learning Outcomes

Through this project, I practiced:

* Real-world data cleaning
* Missing value treatment
* Exploratory data analysis
* Feature importance analysis
* Ordinal vs One-Hot Encoding decisions
* Building preprocessing pipelines
* Training and evaluating regression models

The main focus of this project was developing a strong preprocessing and EDA workflow rather than maximizing model performance through advanced algorithms.
