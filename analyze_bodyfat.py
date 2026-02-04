import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os

# Create output directory for visualizations
if not os.path.exists('figures'):
    os.makedirs('figures')

# Style of plots
sns.set_theme(style="whitegrid")
sns.set_palette("deep")
sns.set_context("notebook")

# Data Load
print("Loading and cleaning data...")
df = pd.read_excel('PercentBodyFat.xlsx')

# Data cleaning
df = df.drop(columns=['Unnamed: 14'])

# Outliers in the target variable
Q1 = df['PercentBodyFat'].quantile(0.25)
Q3 = df['PercentBodyFat'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Potential outliers in PercentBodyFat: values < {lower_bound:.2f} or > {upper_bound:.2f}")

# Analyzing extreme values
outliers = df[(df['PercentBodyFat'] < lower_bound) | (df['PercentBodyFat'] > upper_bound)]
print(f"Number of potential outliers: {len(outliers)}")

# Print extreme low values (near 0 body fat is physiologically impossible)
extreme_low = df[df['PercentBodyFat'] < 3]
print(f"Extreme low values (below 3% body fat):")
print(extreme_low[['PercentBodyFat', 'Weight', 'Height', 'Age']])

# Handle extreme values - remove observations with body fat < 3% (body fat below 3% is not compatible with life)
df_cleaned = df[df['PercentBodyFat'] >= 3]
print(f"Rows remaining after removing extreme values: {len(df_cleaned)}")

# Save the exploratory output to a file
with open('data_analysis_report.txt', 'w') as f:
    f.write("Dataset Information:\n")
    f.write(str(df.dtypes) + "\n\n")
    f.write("Summary Statistics (before cleaning):\n")
    f.write(str(df.describe()) + "\n\n")
    f.write("Summary Statistics (after cleaning):\n")
    f.write(str(df_cleaned.describe()) + "\n\n")

# Exploratory Data Analysis
print("\nPerforming Exploratory Data Analysis...")

# Correlation matrix
plt.figure(figsize=(14, 12))
corr_matrix = df_cleaned.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('figures/correlation_matrix.png')
plt.close()

# Correlation with target
correlation_with_target = corr_matrix['PercentBodyFat'].sort_values(ascending=False)
print("\nCorrelation with PercentBodyFat:")
print(correlation_with_target)

# Visualize correlation with target
plt.figure(figsize=(10, 8))
correlation_with_target[1:].plot(kind='bar')  # Skip the first one which is the correlation with itself
plt.title('Correlation with Body Fat Percentage')
plt.tight_layout()
plt.savefig('figures/correlations_with_target.png')
plt.close()

# Save to report
with open('data_analysis_report.txt', 'a') as f:
    f.write("Correlation with PercentBodyFat:\n")
    f.write(str(correlation_with_target) + "\n\n")

# Identify highly correlated features
threshold = 0.7
high_corr = {}
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname_i = corr_matrix.columns[i]
            colname_j = corr_matrix.columns[j]
            high_corr[f"{colname_i}-{colname_j}"] = corr_matrix.iloc[i, j]

print("\nHighly correlated features (correlation > 0.7):")
print(high_corr)

with open('data_analysis_report.txt', 'a') as f:
    f.write("Highly correlated features (correlation > 0.7):\n")
    f.write(str(high_corr) + "\n\n")

# Pairplot of the most correlated features with target
top_correlated = correlation_with_target.index[:6]  # Top 5 correlations + the target itself
plt.figure(figsize=(12, 10))
sns.pairplot(df_cleaned[top_correlated])
plt.tight_layout()
plt.savefig('figures/pairplot_top_features.png')
plt.close()

# Feature importance analysis
print("\nPerforming feature importance analysis...")

# Prepare data for modeling
X = df_cleaned.drop('PercentBodyFat', axis=1)
y = df_cleaned['PercentBodyFat']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Random Forest Feature Importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance)

# Visualize Random Forest importance
plt.figure(figsize=(10, 6))
rf_importance.plot(kind='bar')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('figures/rf_feature_importance.png')
plt.close()

# Save to report
with open('data_analysis_report.txt', 'a') as f:
    f.write("Random Forest Feature Importance:\n")
    f.write(str(rf_importance) + "\n\n")

# 2. Recursive Feature Elimination
print("\nPerforming Recursive Feature Elimination...")
rfe = RFE(estimator=LinearRegression(), n_features_to_select=5)
rfe = rfe.fit(X_train, y_train)

rfe_selected = X.columns[rfe.support_]
print("Top features selected by RFE:")
print(rfe_selected)

with open('data_analysis_report.txt', 'a') as f:
    f.write("Features selected by RFE:\n")
    f.write(str(rfe_selected.tolist()) + "\n\n")

# 3. Sequential R² Analysis
print("\nPerforming sequential R² analysis...")

def calculate_r2_changes(X_train, X_test, y_train, y_test):
    features = list(X_train.columns)
    n_features = len(features)
    
    # Calculate individual feature R²
    individual_r2 = {}
    for feature in features:
        model = LinearRegression()
        model.fit(X_train[[feature]], y_train)
        y_pred = model.predict(X_test[[feature]])
        individual_r2[feature] = r2_score(y_test, y_pred)
    
    # Sort features by individual R²
    sorted_features = sorted(individual_r2.items(), key=lambda x: x[1], reverse=True)
    ordered_features = [f[0] for f in sorted_features]
    
    # Incremental R² with features added in order of importance
    cumulative_r2 = {}
    previous_r2 = 0
    included_features = []
    
    for feature in ordered_features:
        included_features.append(feature)
        model = LinearRegression()
        model.fit(X_train[included_features], y_train)
        y_pred = model.predict(X_test[included_features])
        current_r2 = r2_score(y_test, y_pred)
        r2_change = current_r2 - previous_r2
        cumulative_r2[feature] = {
            'individual_r2': individual_r2[feature],
            'cumulative_r2': current_r2,
            'r2_change': r2_change
        }
        previous_r2 = current_r2
    
    return cumulative_r2, individual_r2, ordered_features

r2_analysis, individual_r2, ordered_features = calculate_r2_changes(X_train, X_test, y_train, y_test)

# Print R² analysis results
print("\nR² Analysis Results:")
for feature in ordered_features:
    info = r2_analysis[feature]
    print(f"{feature}: Individual R² = {info['individual_r2']:.4f}, "
          f"Cumulative R² = {info['cumulative_r2']:.4f}, "
          f"R² Change = {info['r2_change']:.4f}")

# Visualize the R² changes
r2_changes = pd.Series({f: r2_analysis[f]['r2_change'] for f in ordered_features})
plt.figure(figsize=(12, 6))
r2_changes.plot(kind='bar')
plt.title('R² Change When Adding Each Feature')
plt.ylabel('Change in R²')
plt.tight_layout()
plt.savefig('figures/r2_changes.png')
plt.close()

# Visualize cumulative R²
cumulative_r2_values = [r2_analysis[f]['cumulative_r2'] for f in ordered_features]
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(ordered_features) + 1), cumulative_r2_values, marker='o')
plt.xticks(range(1, len(ordered_features) + 1), ordered_features, rotation=90)
plt.title('Cumulative R² with Features Added in Order of Importance')
plt.ylabel('Cumulative R²')
plt.xlabel('Features Added')
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/cumulative_r2.png')
plt.close()

# Save to report
with open('data_analysis_report.txt', 'a') as f:
    f.write("R² Analysis Results:\n")
    for feature in ordered_features:
        info = r2_analysis[feature]
        f.write(f"{feature}: Individual R² = {info['individual_r2']:.4f}, "
               f"Cumulative R² = {info['cumulative_r2']:.4f}, "
               f"R² Change = {info['r2_change']:.4f}\n")
    f.write("\n")

# Build and evaluate regression models
print("\nBuilding regression models...")

# Build a multiple linear regression model using all features
lm_all = LinearRegression()
lm_all.fit(X_train, y_train)
y_pred_all = lm_all.predict(X_test)
r2_all = r2_score(y_test, y_pred_all)
rmse_all = np.sqrt(mean_squared_error(y_test, y_pred_all))

print(f"\nRegression with all features:")
print(f"R² = {r2_all:.4f}, RMSE = {rmse_all:.4f}")

# Build a model with top 5 features from R² analysis
top5_features = ordered_features[:5]
lm_top5 = LinearRegression()
lm_top5.fit(X_train[top5_features], y_train)
y_pred_top5 = lm_top5.predict(X_test[top5_features])
r2_top5 = r2_score(y_test, y_pred_top5)
rmse_top5 = np.sqrt(mean_squared_error(y_test, y_pred_top5))

print(f"\nRegression with top 5 features ({', '.join(top5_features)}):")
print(f"R² = {r2_top5:.4f}, RMSE = {rmse_top5:.4f}")

# Build a model using the features selected by RFE
lm_rfe = LinearRegression()
lm_rfe.fit(X_train[rfe_selected], y_train)
y_pred_rfe = lm_rfe.predict(X_test[rfe_selected])
r2_rfe = r2_score(y_test, y_pred_rfe)
rmse_rfe = np.sqrt(mean_squared_error(y_test, y_pred_rfe))

print(f"\nRegression with RFE selected features ({', '.join(rfe_selected)}):")
print(f"R² = {r2_rfe:.4f}, RMSE = {rmse_rfe:.4f}")

# Compare models
model_comparison = pd.DataFrame({
    'Model': ['All Features', 'Top 5 Features', 'RFE Features'],
    'Features': [len(X.columns), 5, 5],
    'R²': [r2_all, r2_top5, r2_rfe],
    'RMSE': [rmse_all, rmse_top5, rmse_rfe]
})

print("\nModel Comparison:")
print(model_comparison)

# Visualize model comparison
plt.figure(figsize=(10, 6))
model_comparison.plot(x='Model', y='R²', kind='bar', ax=plt.gca())
plt.title('R² Comparison Across Models')
plt.tight_layout()
plt.savefig('figures/model_r2_comparison.png')
plt.close()

# Save to report
with open('data_analysis_report.txt', 'a') as f:
    f.write("Model Comparison:\n")
    f.write(str(model_comparison) + "\n\n")

# Get the coefficients and equation for the best model
best_model_name = model_comparison.iloc[model_comparison['R²'].argmax()]['Model']
print(f"\nThe best model based on R² is: {best_model_name}")

# Determine which model is best
if best_model_name == 'All Features':
    best_model = lm_all
    best_features = X.columns
elif best_model_name == 'Top 5 Features':
    best_model = lm_top5
    best_features = top5_features
else:  # 'RFE Features'
    best_model = lm_rfe
    best_features = rfe_selected

# Get coefficients
coef_df = pd.DataFrame({
    'Feature': best_features,
    'Coefficient': best_model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\nCoefficients of the best model:")
print(coef_df)

# Print the regression equation
intercept = best_model.intercept_
equation = f"PercentBodyFat = {intercept:.4f}"
for index, row in coef_df.iterrows():
    if row['Coefficient'] >= 0:
        equation += f" + {row['Coefficient']:.4f} × {row['Feature']}"
    else:
        equation += f" - {abs(row['Coefficient']):.4f} × {row['Feature']}"

print("\nRegression Equation:")
print(equation)

# Save to report
with open('data_analysis_report.txt', 'a') as f:
    f.write("Best Model: " + best_model_name + "\n")
    f.write("Coefficients of the best model:\n")
    f.write(str(coef_df) + "\n\n")
    f.write("Regression Equation:\n")
    f.write(equation + "\n\n")

# Visualize actual vs predicted for the best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_model.predict(X_test[best_features]), alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Body Fat Percentage')
plt.ylabel('Predicted Body Fat Percentage')
plt.title(f'Actual vs Predicted Body Fat Percentage ({best_model_name})')
plt.tight_layout()
plt.savefig('figures/actual_vs_predicted.png')
plt.close()

# Cross-validation for the best model
cv_scores = cross_val_score(best_model, X[best_features], y, cv=5, scoring='r2')
print(f"\nCross-validation R² scores for the best model:")
print(f"Mean R² = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")
print(f"Individual scores: {cv_scores}")

# Save to report
with open('data_analysis_report.txt', 'a') as f:
    f.write("Cross-validation R² scores for the best model:\n")
    f.write(f"Mean R² = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}\n")
    f.write(f"Individual scores: {cv_scores}\n\n")

# Additional analysis using statsmodels for detailed statistical metrics
print("\nDetailed statistical analysis of the best model using statsmodels:")
X_sm = sm.add_constant(X[best_features])
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.summary())

# Save to report
with open('data_analysis_report.txt', 'a') as f:
    f.write("Detailed Statistical Analysis:\n")
    f.write(str(model_sm.summary()) + "\n\n")

# Final summary output
print("\nAnalysis complete! Results have been saved to 'data_analysis_report.txt' and figures.")
print("Key findings:")
print(f"1. Most important features for predicting body fat: {', '.join(best_features[:3])}")
print(f"2. Best model performance: R² = {model_comparison['R²'].max():.4f}")
print(f"3. Final regression equation: {equation}") 