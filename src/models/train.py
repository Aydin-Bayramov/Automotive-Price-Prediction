import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

save_dir = os.path.join(os.path.dirname(__file__), "../../models", "trained_models")
os.makedirs(save_dir, exist_ok=True)

def load_data():
    return pd.read_csv("data/processed/prepared_turbo_az.csv")

def split_data(df):
    X = df.drop("Price", axis=1)
    y = df["Price"]
    return train_test_split(X, y, test_size=0.2, random_state=0)

def regression_metrics(y_true, y_pred):
    """Calculates and prints regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"      MAE: {mae:.2f}")
    print(f"      MSE: {mse:.2f}")
    print(f"      RMSE: {rmse:.2f}")
    print(f"      R²: {r2:.4f}")

def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("=== Linear Regression (Training Set) ===")
    regression_metrics(y_train, y_train_pred)
    print('-' * 50)
    print("=== Linear Regression (Test Set) ===")
    regression_metrics(y_test, y_test_pred)
    
    model_path = os.path.join(save_dir, "linear_regression.pkl")
    joblib.dump(model, model_path)

def train_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsRegressor()

    param_grid = {
        'n_neighbors': [4, 5, 6],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    print(f"Best Parameters: {grid_search.best_params_}")
    print("\n=== K-Nearest Neighbors (Training Set) ===")
    regression_metrics(y_train, y_train_pred)
    print('-' * 50)
    print("=== K-Nearest Neighbors (Test Set) ===")
    regression_metrics(y_test, y_test_pred)

    model_path = os.path.join(save_dir, "knn.pkl")
    joblib.dump(best_model, model_path)

def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(random_state=0)

    param_grid = {
        "max_depth": [5, 10, 15],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [2, 5, 10] 
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    print(f"Best Parameters: {grid_search.best_params_}")
    print("\n=== Decision Tree Regression (Training Set) ===")
    regression_metrics(y_train, y_train_pred)
    print('-' * 50)
    print("=== Decision Tree Regression (Test Set) ===")
    regression_metrics(y_test, y_test_pred)

    model_path = os.path.join(save_dir, "decision_tree.pkl")
    joblib.dump(best_model, model_path)

def train_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=0)

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [15, 20, None],
        "min_samples_split": [2, 5, 8],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", "log2"]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    print(f"Best Parameters: {grid_search.best_params_}")
    print("\n=== Random Forest (Training Set) ===")
    regression_metrics(y_train, y_train_pred)
    print('-' * 50)
    print("=== Random Forest (Test Set) ===")
    regression_metrics(y_test, y_test_pred)

    model_path = os.path.join(save_dir, "random_forest.pkl")
    joblib.dump(best_model, model_path)

def train_stacking_regressor(X_train, X_test, y_train, y_test):
    # Load pre-trained models
    base_models = [
        #('lr', joblib.load(os.path.join(save_dir, "linear_regression.pkl"))),
        ('knn', joblib.load(os.path.join(save_dir, "knn.pkl"))),
        ('dt', joblib.load(os.path.join(save_dir, "decision_tree.pkl"))),
        ('rf', joblib.load(os.path.join(save_dir, "random_forest.pkl")))
    ]

    meta_model = LinearRegression()

    # Create stacking regressor
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)
    stacking_model.fit(X_train, y_train)

    y_train_pred = stacking_model.predict(X_train)
    y_test_pred = stacking_model.predict(X_test)

    print("\n=== Stacking Regressor (Training Set) ===")
    regression_metrics(y_train, y_train_pred)
    print('-' * 50)
    print("=== Stacking Regressor (Test Set) ===")
    regression_metrics(y_test, y_test_pred)

    model_path = os.path.join(save_dir, "stacking_regressor.pkl")
    joblib.dump(stacking_model, model_path)

def main():
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    # Train individual models
    train_linear_regression(X_train, X_test, y_train, y_test)
    train_knn(X_train, X_test, y_train, y_test)
    train_decision_tree(X_train, X_test, y_train, y_test)
    train_random_forest(X_train, X_test, y_train, y_test)

    # Train stacking regressor using pre-trained models
    train_stacking_regressor(X_train, X_test, y_train, y_test)
       
if __name__ == "__main__":
    main()