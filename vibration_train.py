from datetime import datetime, timedelta
from model import *
import pandas as pd  # type: ignore
import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore
import matplotlib.pyplot as plt
from predict_detail import main as predict_detail


def prepare_data(values):
    """
    Prepare time series data in monthly format
    """
    df = pd.DataFrame(values, columns=["timestamp", "value"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    # Resample to monthly data, taking the mean
    df = df.resample("M").mean()
    
    return df


def manual_seasonal_decompose(data, period=12):
    """
    Custom seasonal decomposition to handle missing values
    """
    # Interpolate missing values
    data_filled = data.interpolate()
    
    # Trend component using moving average
    trend = data_filled.rolling(window=period, center=True, min_periods=1).mean()
    
    # Seasonal component
    detrended = data_filled - trend
    seasonal = detrended.groupby(detrended.index.month).mean()
    seasonal = pd.Series(
        [seasonal[month] for month in data_filled.index.month], 
        index=data_filled.index
    )
    
    # Residual
    residual = data_filled - (trend + seasonal)
    
    return trend, seasonal, residual


def prepare_data_with_decomposition(values):
    """
    Prepare data with decomposition
    """
    df = prepare_data(values)
    
    # Use custom decomposition method
    trend, seasonal, residual = manual_seasonal_decompose(df["value"])

    # Remove NaN values
    valid_idx = ~(trend.isna() | seasonal.isna() | residual.isna())
    trend = trend[valid_idx]
    seasonal = seasonal[valid_idx]
    residual = residual[valid_idx]

    df_decomposed = pd.DataFrame(
        {"trend": trend, "seasonal": seasonal, "residual": residual}
    )

    return df_decomposed


def find_best_parameters(data):
    """
    Find best ARIMA parameters
    Simplified for monthly data
    """
    best_aic = float("inf")
    best_params = None

    # Reduced parameter ranges for monthly data
    p_values = range(0, 2)
    d_values = range(0, 2)
    q_values = range(0, 2)

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(data, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_params = (p, d, q)
                except:
                    continue

    return best_params or (1, 1, 1)  # Default parameters if optimization fails


def train_arima_with_decomposition(data):
    """
    Train ARIMA model for each component
    """
    models = {}

    for component in ["trend", "seasonal", "residual"]:
        best_params = find_best_parameters(data[component])
        print(f"Best parameters for {component}: {best_params}")

        model = ARIMA(data[component], order=best_params)
        fitted_model = model.fit()
        models[component] = fitted_model

    return models


def forecast_with_decomposition(models, steps=12):  # 12 steps = 1 year
    """
    Create prediction by combining forecasts of each component
    """
    forecasts = {}

    # Forecast for each component
    for component, model in models.items():
        forecast = model.forecast(steps=steps)
        forecasts[component] = forecast

    # Combine forecasts
    final_forecast = forecasts["trend"] + forecasts["seasonal"] + forecasts["residual"]

    return final_forecast


def plot_forecast(actual_data, forecast_df):
    """
    Visualize original data and 1-year forecast
    """
    plt.figure(figsize=(20, 10))

    # Plot actual data
    plt.plot(
        actual_data.index, actual_data.values, label="Historical Data", color="blue"
    )

    # Plot forecast
    plt.plot(
        forecast_df.index,
        forecast_df["forecast"],
        label="Forecast",
        color="red",
        linestyle="--",
    )

    # Plot confidence interval
    plt.fill_between(
        forecast_df.index,
        forecast_df["lower_ci"],
        forecast_df["upper_ci"],
        color="red",
        alpha=0.1,
        label="95% Confidence Interval",
    )

    plt.title("1-Year Monthly Forecast with ARIMA")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Rotate date labels
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Annotate forecast values
    for date, value in forecast_df["forecast"].items():
        plt.annotate(
            f"{value:.2f}",
            (date, value),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
        )

    plt.show()
    plt.savefig("monthly_forecast.png")


def main(part_id, features_id):
    # Get data
    values = get_values(part_id, features_id)

    steps = 6  # 12 months forecast
    periods = steps + 1

    # Prepare data with decomposition
    df_decomposed = prepare_data_with_decomposition(values)

    # Train model for each component
    models = train_arima_with_decomposition(df_decomposed)

    # Create forecast for 12 months
    forecast = forecast_with_decomposition(models, steps=steps)
    print(forecast)

    # Create DataFrame for forecast results
    last_date = df_decomposed.index[-1]
    forecast_index = pd.date_range(start=last_date, periods=periods, freq="M")[1:]
    forecast_df = pd.DataFrame(
        {
            "forecast": forecast,
            "features_id": features_id,
            "part_id": part_id,
            "lower_ci": forecast - 2 * forecast.std(),
            "upper_ci": forecast + 2 * forecast.std(),
        },
        index=forecast_index,
    )

    # Save predictions
    save_predictions_to_db(forecast_df)
    predict_detail(part_id)

    # Return forecast results
    return forecast_df


if __name__ == "__main__":
    main("0d25b4e6-972b-4c40-b964-d17e09a3d3fa", "9dcb7e40-ada7-43eb-baf4-2ed584233de7")