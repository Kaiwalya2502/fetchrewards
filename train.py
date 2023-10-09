import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.Series: The 'Receipt_Count' column from the loaded CSV file.
    """
    data = pd.read_csv(file_path)
    return data['Receipt_Count']

def plot_and_save_timeseries(data, save_path='plots/time_series_plot.png'):
    """
    Plot the time series data and save the plot as an image file.

    Parameters:
    - data (pd.Series): The time series data to plot.
    - save_path (str, optional): The path where the plot image should be saved. Defaults to 'time_series_plot.png'.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Observed')
    plt.title('Time Series Data')
    plt.xlabel('Date')
    plt.ylabel('Receipt_Count')
    plt.legend()
    plt.savefig(save_path)
    # plt.show()

def fit_arima_model(data, order = (1, 1, 1)):
    """
    Fits an ARIMA model to the time series data.

    Parameters:
    - data (pd.Series): The time series data to model.
    - order (tuple, optional): The (p, d, q) order of the ARIMA model. Defaults to (1, 1, 1).

    Returns:
    - ARIMAResultsWrapper: The fitted ARIMA model.
    """
    model = ARIMA(data, order = order)
    model_fit = model.fit()
    return model_fit


def forecast_and_plot(data, model_fit, steps = 30, alpha = 0.05, save_path = 'plots/forecast_plot.png'):
    """
    Forecast future values using an ARIMA model and plot the results.

    Parameters:
    - data (pd.Series): The time series data.
    - model_fit (ARIMAResultsWrapper): The fitted ARIMA model.
    - steps (int, optional): The number of future steps to forecast. Defaults to 30.
    - alpha (float, optional): The significance level for the confidence interval (0 < alpha < 1). Defaults to 0.05.
    - save_path (str, optional): The path where the plot image should be saved. Defaults to 'forecast_plot.png'.

    Returns:
    - pd.Series: The forecasted values.
    """
    prediction = model_fit.get_prediction(end = len(data)-1)
    forecast = model_fit.get_forecast(steps = steps)
    
    # Confidence intervals
    pred_confidence_interval = prediction.conf_int(alpha = alpha)
    forecast_confidence_interval = forecast.conf_int(alpha = alpha)
    
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Observed')
    plt.plot(prediction.predicted_mean.index[-10:], prediction.predicted_mean.values[-10:], color='green', label='Predicted')
    plt.plot(forecast.predicted_mean.index, forecast.predicted_mean.values, color='red', label='Forecast')
    plt.fill_between(pred_confidence_interval.index[-10:], 
                     pred_confidence_interval.iloc[-10:, 0], 
                     pred_confidence_interval.iloc[-10:, 1], color='palegreen', alpha=0.5)
    plt.fill_between(forecast_confidence_interval.index, 
                     forecast_confidence_interval.iloc[:, 0], 
                     forecast_confidence_interval.iloc[:, 1], color='pink', alpha=0.5)
    plt.title('ARIMA Forecast with 95% Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(save_path)
    # plt.show()
    return forecast.predicted_mean

def linear_regression_forecast(data, steps=30, save_path='plots/lr_forecast_plot.png'):
    """
    Forecast future values using a simple linear regression model and plot the results.

    Parameters:
    - data (pd.Series): The time series data.
    - steps (int, optional): The number of future steps to forecast. Defaults to 30.
    - save_path (str, optional): The path where the plot image should be saved. Defaults to 'lr_forecast_plot.png'.

    Returns:
    - np.ndarray: The forecasted values.
    """
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.values.reshape(-1, 1)
    theta = linear_regression(X, y)
    
    X_all = np.arange(len(data) + steps).reshape(-1, 1)
    y_all = predict_with_lr(theta, X_all)
    y_hat = predict_with_lr(theta, X)
    residuals = y - y_hat
    mse = np.mean(residuals**2)
    se = np.sqrt(mse)
    ci = 1.96 * se  # 95% confidence interval
    lower_bound = y_all - ci
    upper_bound = y_all + ci
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data.values, label='Observed')
    plt.plot(np.arange(len(data) + steps), y_all, label='Linear Fit and Forecast', color='green')
    plt.fill_between(np.arange(len(data) + steps), lower_bound.flatten(), upper_bound.flatten(), color='palegreen', alpha=0.5)
    plt.title('Linear Regression Forecast with 95% Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(save_path)
    # plt.show()
    
    return y_all[-steps:]


def linear_regression(X, y):
    """
    Fit a simple linear regression model.

    Parameters:
    - X (np.ndarray): The independent variable(s).
    - y (np.ndarray): The dependent variable.

    Returns:
    - np.ndarray: The parameters (theta) of the fitted model.
    """
    X_b = np.c_[np.ones((len(X), 1)), X] 
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    with open("models/lr_model.pkl", 'wb') as f:
        pickle.dump(theta_best, f)
    return theta_best

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


# Usage example in the script:
if __name__ == "__main__":
    file_path = sys.argv[1]
    data = load_data(file_path)
    
    plot_and_save_timeseries(data)
    
    model_fit = fit_arima_model(data, (10, 1, 0))
    print(model_fit.summary())
    
    forecast = forecast_and_plot(data, model_fit)
    print(f"ARIMA Forecasts:\n{int(sum(forecast))}")
    
    lr_forecast = linear_regression_forecast(data)
    print(f"Linear Regression Forecasts:\n{int(sum(lr_forecast))}")
