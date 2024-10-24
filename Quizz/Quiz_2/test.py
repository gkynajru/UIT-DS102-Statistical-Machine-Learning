import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# Load the train and test data
train_data = np.load('public_data//train//train.npz')
test_data = np.load('public_data//test//test.npz')

# Extract training and test data
X_train = train_data['X_train']
y_train = train_data['y_train']
X_test = test_data['X_test']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the RandomForestRegressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Evaluate the model using cross-validation (5-fold CV)
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')

# Calculate mean and standard deviation of MSE from cross-validation
rf_mean_mse = np.mean(-rf_cv_scores)
rf_std_mse = np.std(-rf_cv_scores)

print(f"RandomForest Mean MSE: {rf_mean_mse}")
print(f"RandomForest Std MSE: {rf_std_mse}")

# Fit the model on the full training data
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test_scaled)

# Save the predictions as a DataFrame
pred_df = pd.DataFrame({'y': y_pred})

# Save the predictions to a JSON file
pred_df.to_json('predictions_test.json', orient='records', lines=True)

# Optionally, compute MSE on the test data if true labels are available (for demonstration, use the same y_train)
# Note: You would typically compare y_pred with true test labels (y_test), but since they are not provided, this is a placeholder.
# mse_test = mean_squared_error(y_train, y_pred)  # Replace y_train with true y_test if available
# print(f"Test MSE: {mse_test}")
