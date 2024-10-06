import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('friedman.csv')

# Features are Input1 to Input5, and the target is Output
X = data[['Input1', 'Input2', 'Input3', 'Input4', 'Input5']]  # Features
y = data['Output']  # Target variable

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)  # Adjust alpha for Ridge regularization
lasso_model = Lasso(alpha=1.0)  # Adjust alpha for LASSO regularization

# Fit the models on the training data
linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_linear = linear_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)
y_pred_lasso = lasso_model.predict(X_test)

# Calculate MSE and MAE for Linear, Ridge, and LASSO regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

# Print the results
print("Linear Regression: MSE =", mse_linear, ", MAE =", mae_linear)
print("Ridge Regression: MSE =", mse_ridge, ", MAE =", mae_ridge)
print("LASSO Regression: MSE =", mse_lasso, ", MAE =", mae_lasso)

# Print the coefficients of all models
print("Linear Regression Coefficients:", linear_model.coef_)
print("Ridge Regression Coefficients:", ridge_model.coef_)
print("LASSO Regression Coefficients:", lasso_model.coef_)
