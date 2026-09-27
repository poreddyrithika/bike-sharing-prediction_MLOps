import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_training():
    print("Starting Model Training...")

    # 1. Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze("columns")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze("columns")

    # 2. Baseline models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    trained_models = {}
    results = []
    for name, model in models.items():
        print("Training:", name)
        model.fit(X_train, y_train)
        trained_models[name] = model

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2})

    results_df = pd.DataFrame(results)

    # 3. Hyperparameter tuning for Random Forest (TimeSeriesSplit + RandomizedSearchCV)
    print("Tuning Random Forest via RandomizedSearchCV...")
    tscv = TimeSeriesSplit(n_splits=5)
    rf_parameters = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 1.0],
    }
    rf_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions=rf_parameters,
        n_iter=20,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    print("Best Random Forest parameters:", rf_search.best_params_)

    rf_preds = rf_search.predict(X_test)
    tuned_result = {
        "Model": "Tuned Random Forest",
        "MAE": mean_absolute_error(y_test, rf_preds),
        "MSE": mean_squared_error(y_test, rf_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, rf_preds)),
        "R2 Score": r2_score(y_test, rf_preds),
    }
    final_comparison = pd.concat([results_df, pd.DataFrame([tuned_result])], ignore_index=True)
    trained_models["Tuned Random Forest"] = rf_search

    # 4. Select best model by RMSE
    best_model_name = final_comparison.loc[final_comparison["RMSE"].idxmin(), "Model"]
    print("Best model based on RMSE:", best_model_name)
    final_model = trained_models[best_model_name]

    # 5. Save artifacts
    joblib.dump(final_model, "models/final_model.pkl")
    joblib.dump(best_model_name, "models/best_model_name.pkl")
    final_comparison.to_csv("outputs/model_comparison_results.csv", index=False)

    print("Model training complete and saved to disk!")


if __name__ == "__main__":
    run_training()
