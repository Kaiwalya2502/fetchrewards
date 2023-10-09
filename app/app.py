from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

def predict_with_lr(theta, X_new):
    """
    Predict values using a fitted linear regression model.

    Parameters:
    - theta (np.ndarray): The parameters of the linear regression model.
    - X_new (np.ndarray): The values of the independent variable(s) for which to predict the dependent variable.

    Returns:
    - np.ndarray: The predicted values.
    """
    X_new_b = np.c_[np.ones((len(X_new), 1)), X_new] 
    return X_new_b.dot(theta)

@app.route('/get_original_data')
def get_original_data():
    df = pd.read_csv('data/data_daily.csv')  # Your CSV data path
    return jsonify(x=df['# Date'].tolist(), y=df['Receipt_Count'].tolist())

@app.route('/predict')
def predict():
    days = int(request.args.get('days'))
    df = pd.read_csv('data/data_daily.csv')  # Your CSV data path
    
    with open('models/lr_model.pkl', 'rb') as f:
        theta = pickle.load(f)

    # Preparing data
    x = np.arange(len(df)).reshape(-1, 1)

    # Predicting
    x_future = np.arange(len(df), len(df) + days).reshape(-1, 1)
    y_pred = predict_with_lr(theta, x_future)

    x_all = np.concatenate((x, x_future)).flatten()
    y_all = np.concatenate((df['Receipt_Count'].values, y_pred.flatten()))
    total = int(sum(y_all[-days:]))
    # Converting dates if your 'Date' column is in datetime format
    dates = pd.date_range(start=df['# Date'].iloc[0], periods=len(x_all), freq='D')

    return jsonify(x=dates.strftime('%Y-%m-%d').tolist(), y=y_all.tolist(), total = total)

if __name__ == "__main__":
    app.run(debug=True)
