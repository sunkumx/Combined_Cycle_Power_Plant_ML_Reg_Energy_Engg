from sklearn.metrics import mean_squared_error

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Test RSME: {rmse}")
    return rmse