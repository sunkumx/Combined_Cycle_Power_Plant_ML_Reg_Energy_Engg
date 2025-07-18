from sklearn.metrics import root_mean_squared_error

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"Test RSME: {rmse}")
    return rmse