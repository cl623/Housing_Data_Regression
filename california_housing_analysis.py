import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the California housing dataset
print("Loading California housing dataset...")
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

print(f"Dataset shape: {X.shape}")
print(f"Features: {california_housing.feature_names}")
print(f"Target variable: {california_housing.target_names[0]}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale the features (important for Ridge regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*50)
print("LINEAR REGRESSION MODEL")
print("="*50)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Make predictions
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)

# Calculate RMSE
lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))

print(f"Linear Regression - Training RMSE: {lr_train_rmse:.4f}")
print(f"Linear Regression - Test RMSE: {lr_test_rmse:.4f}")

# Display coefficients
print(f"\nLinear Regression coefficients:")
for feature, coef in zip(california_housing.feature_names, lr_model.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"Intercept: {lr_model.intercept_:.4f}")

print("\n" + "="*50)
print("RIDGE REGRESSION MODELS")
print("="*50)

# Test different alpha values for Ridge regression
alpha_values = [0.1, 1, 10, 100]
ridge_results = []

for alpha in alpha_values:
    print(f"\nTraining Ridge model with alpha = {alpha}")
    
    # Train Ridge model
    ridge_model = Ridge(alpha=alpha, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    ridge_train_pred = ridge_model.predict(X_train_scaled)
    ridge_test_pred = ridge_model.predict(X_test_scaled)
    
    # Calculate RMSE
    ridge_train_rmse = np.sqrt(mean_squared_error(y_train, ridge_train_pred))
    ridge_test_rmse = np.sqrt(mean_squared_error(y_test, ridge_test_pred))
    
    ridge_results.append({
        'alpha': alpha,
        'train_rmse': ridge_train_rmse,
        'test_rmse': ridge_test_rmse,
        'model': ridge_model
    })
    
    print(f"  Training RMSE: {ridge_train_rmse:.4f}")
    print(f"  Test RMSE: {ridge_test_rmse:.4f}")
    
    # Display coefficients
    print(f"  Coefficients:")
    for feature, coef in zip(california_housing.feature_names, ridge_model.coef_):
        print(f"    {feature}: {coef:.4f}")
    print(f"  Intercept: {ridge_model.intercept_:.4f}")

print("\n" + "="*50)
print("PERFORMANCE COMPARISON")
print("="*50)

# Compare all models
print(f"{'Model':<25} {'Training RMSE':<15} {'Test RMSE':<15}")
print("-" * 55)
print(f"{'Linear Regression':<25} {lr_train_rmse:<15.4f} {lr_test_rmse:<15.4f}")

for result in ridge_results:
    print(f"{'Ridge (α=' + str(result['alpha']) + ')':<25} {result['train_rmse']:<15.4f} {result['test_rmse']:<15.4f}")

# Find best performing model
best_ridge = min(ridge_results, key=lambda x: x['test_rmse'])
print(f"\nBest Ridge model: alpha = {best_ridge['alpha']}")

if best_ridge['test_rmse'] < lr_test_rmse:
    improvement = lr_test_rmse - best_ridge['test_rmse']
    print(f"Ridge regularization improves RMSE by {improvement:.4f}")
    print("Regularization is beneficial for this dataset!")
else:
    print("Linear Regression performs better than Ridge regression")
    print("Regularization does not improve performance for this dataset")

# Create visualization
plt.figure(figsize=(12, 5))

# Plot 1: RMSE comparison
plt.subplot(1, 2, 1)
models = ['Linear'] + [f'Ridge (α={r["alpha"]})' for r in ridge_results]
test_rmses = [lr_test_rmse] + [r['test_rmse'] for r in ridge_results]

plt.bar(models, test_rmses, color=['blue'] + ['green']*len(ridge_results))
plt.title('Test RMSE Comparison')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot 2: Training vs Test RMSE
plt.subplot(1, 2, 2)
train_rmses = [lr_train_rmse] + [r['train_rmse'] for r in ridge_results]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, train_rmses, width, label='Training RMSE', alpha=0.8)
plt.bar(x + width/2, test_rmses, width, label='Test RMSE', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Training vs Test RMSE')
plt.xticks(x, models, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nAnalysis complete! Check the plots above for visual comparison.") 