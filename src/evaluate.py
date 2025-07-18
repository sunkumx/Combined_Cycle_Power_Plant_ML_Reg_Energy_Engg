import os
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

def evaluate(model, X_test, y_test, model_name="RandomForestRegressor", output_dir="model-output"):
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"model_output_{timestamp}.txt"
    output_path = os.path.join(output_dir, filename)

    model_details = str(model.get_params())

    output = (
        f"Evaluation Timestamp: {timestamp}\n"
        f"Model Type: {model_name}\n"
        f"Hyperparameters:\n{model_details}\n\n"
        f"Model Evaluation Metrics:\n"
        f"Mean Squared Error (MSE): {mse:.4f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
        f"Mean Absolute Error (MAE): {mae:.4f}\n"
        f"R² Score: {r2:.4f}\n"
    )

    print(output)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(output)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
