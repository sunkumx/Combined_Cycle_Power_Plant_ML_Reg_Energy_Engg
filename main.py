from src import data_loader, preprocess, train, evaluate

#Load the dataset
df = data_loader.load_data("CCPP.csv")

#Preprocessing the data
X_train, X_test, y_train, y_test, scaler = preprocess.preprocess(df)

#Train the data
model = train.train_model(X_train, y_train)

#Evaluating the model
evaluate.evaluate(model, X_test, y_test)

#saving the model
train.save_model(model, "model/model.joblib")
