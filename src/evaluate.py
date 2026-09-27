import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_evaluation():
    print("Starting Model Evaluation...")

    # 1. Load test data and the trained model
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze("columns")
    final_model = joblib.load("models/final_model.pkl")
    best_model_name = joblib.load("models/best_model_name.pkl")

    # 2. Make predictions
    predictions = final_model.predict(X_test)

    # 3. Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Evaluation Report ---")
    print(f"Best Model: {best_model_name}")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")
    print("--------------------------------\n")

    # 4. Error analysis on a random sample
    random_sample = X_test.sample(n=min(5, len(X_test)), random_state=42)
    actual_values = y_test.loc[random_sample.index]
    predicted_values = final_model.predict(random_sample)
    comparison = pd.DataFrame({
        "Actual": actual_values.values,
        "Predicted": predicted_values.round(),
    })
    comparison["Difference"] = comparison["Actual"] - comparison["Predicted"]
    comparison.to_csv("outputs/sample_predictions.csv", index=False)

    print("Evaluation complete and outputs saved!")


if __name__ == "__main__":
    run_evaluation()
