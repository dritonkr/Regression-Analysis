Body Fat Percentage Estimation Using Regression Models

📌 Project Overview

This project focuses on estimating body fat percentage using simple, easily obtainable physical measurements. Since directly measuring body fat often requires specialized and expensive equipment, this project explores how accurately body fat percentage can be predicted using basic measurements collected with a scale and measuring tape.

A regression-based machine learning approach is used to model the relationship between body fat percentage and several physical indicators.

🎯 Objective

- Analyze the relationship between body fat percentage and physical measurements

- Build a regression model to predict body fat percentage

- Use only non-invasive, easily measurable features

- Improve data reliability by removing physiologically implausible values

📊 Dataset Description

The dataset contains measurements from 252 individuals.

Target Variable:

- PercentBodyFat – Body fat percentage

Features:

- Age

- Weight

- Height

- Body circumference measurements:

  - Neck

  - Chest

  - Abdomen

  - Hip

  - Thigh

  - Knee

  - Ankle

  - Biceps

  - Forearm

  - Wrist

Total variables: 15

🔍 Data Understanding & Preprocessing
Exploratory Data Analysis (EDA)

Initial analysis revealed:

- 252 observations and 15 variables

- Presence of extreme body fat percentage values, including values close to 0%, which are physiologically unrealistic

Data Cleaning

- Removed observations with body fat percentage < 3%

- Final dataset size after cleaning: 249 observations

Distribution

- Body fat percentage follows an approximately normal distribution after cleaning

- Minor skewness remains, but the data is suitable for regression modeling

🧠 Modeling Approach

- A regression model was developed to predict body fat percentage

- Inputs include age, weight, height, and body circumference measurements

- The model learns patterns between physical indicators and body fat percentage


🛠 Tools & Technologies

- Python

- Pandas & NumPy

- Matplotlib / Seaborn (EDA & visualization)

- Scikit-learn (regression modeling)

📈 Results

The model demonstrates that body fat percentage can be reasonably estimated using simple physical measurements, highlighting the strong relationship between body circumferences (especially abdominal measurements) and body fat levels.


🚀 Future Improvements

- Experiment with additional regression models

- Feature selection or dimensionality reduction

- Hyperparameter tuning

- Model comparison and validation

- Deployment as a simple web or mobile app
