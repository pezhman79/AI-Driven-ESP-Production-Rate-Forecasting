import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# Load and prepare the data
df = pd.read_excel("article\\esp.xlsx")
df.columns = list(range(df.shape[1]))

X = df[[0,1,2,3,4,5,6,7,8]].values
Y = df[[9]].values

# Split the data into train, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

target_scaler = StandardScaler()
Y_train = target_scaler.fit_transform(Y_train.reshape(-1, 1)).flatten()
Y_val = target_scaler.transform(Y_val.reshape(-1, 1)).flatten()
Y_test = target_scaler.transform(Y_test.reshape(-1, 1)).flatten()

# Hyperparameter tuning with RandomizedSearchCV
param_distributions = {
    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50), (50,100,50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [1000, 2000, 5000]
}

def negative_mse(estimator, X, y):
    return -mean_squared_error(y, estimator.predict(X))

random_search = RandomizedSearchCV(
    MLPRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,
    cv=5,
    scoring=negative_mse,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, Y_train)
print("Best parameters:", random_search.best_params_)

# Create and train the ANN model with best parameters
model = MLPRegressor(**random_search.best_params_, random_state=42)
model.fit(X_train, Y_train)

# Predict the values
Y_train_pred = model.predict(X_train)
Y_val_pred = model.predict(X_val)
Y_test_pred = model.predict(X_test)

# Calculate MSE for training, validation, and test data
mse_train = mean_squared_error(Y_train, Y_train_pred)
mse_val = mean_squared_error(Y_val, Y_val_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)

print("Mean squared error for training data:", mse_train)
print("Mean squared error for validation data:", mse_val)
print("Mean squared error for test data:", mse_test)

# Calculate SCC for training, validation, and test data
scc_train, _ = spearmanr(Y_train_pred, Y_train)
scc_val, _ = spearmanr(Y_val_pred, Y_val)
scc_test, _ = spearmanr(Y_test_pred, Y_test)

print("Spearman's rank correlation coefficient for training data:", scc_train)
print("Spearman's rank correlation coefficient for validation data:", scc_val)
print("Spearman's rank correlation coefficient for test data:", scc_test)

# Calculate R-squared for training, validation, and test data
r2_train = r2_score(Y_train, Y_train_pred)
r2_val = r2_score(Y_val, Y_val_pred)
r2_test = r2_score(Y_test, Y_test_pred)

print("R-squared for training data:", r2_train)
print("R-squared for validation data:", r2_val)
print("R-squared for test data:", r2_test)

# Feature importance using permutation importance
perm_importance = permutation_importance(model, X_test, Y_test, n_repeats=10, random_state=42)

# Sort and display feature importances
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], xerr=perm_importance.importances_std[sorted_idx])
plt.yticks(range(len(sorted_idx)), df.columns[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance (Permutation)")
plt.show()