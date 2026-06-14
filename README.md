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

## Model

### Linear Regression

A baseline Linear Regression model was trained after preprocessing.

### Results

| Metric | Score  |
| ------ | ------ |
| R²     | 0.883  |
| RMSE   | 30,656 |
| MAE    | 18,079 |

### Interpretation

The model explains approximately **88.3%** of the variance in house prices while maintaining reasonable prediction error, demonstrating the effectiveness of the preprocessing strategy.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn


---

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
