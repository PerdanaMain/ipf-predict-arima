from datetime import datetime, timedelta
from model import *
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from predict_detail import main as predict_detail
import os
import pickle


def prepare_data(values):
    """
    Menyiapkan data time series dalam format yang sesuai
    """
    df = pd.DataFrame(values, columns=["timestamp", "value"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df.resample("h").mean()  # Resampling per jam
    df = df.interpolate(method="linear")
    return df


def decompose_timeseries(data):
    """
    Melakukan dekomposisi time series menjadi komponen trend, seasonal, dan residual
    """
    decomposition = seasonal_decompose(data, period=24)  # period=24 untuk data per jam
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    return trend, seasonal, residual


def prepare_data_with_decomposition(values):
    """
    Menyiapkan data dengan dekomposisi
    """
    df = prepare_data(values)
    trend, seasonal, residual = decompose_timeseries(df["value"])

    # Hapus NaN values
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
    Mencari parameter terbaik untuk model ARIMA
    """
    best_aic = float("inf")
    best_params = None

    # Parameter ranges
    p_values = range(0, 3)  # Reduced range for faster computation
    d_values = range(0, 2)
    q_values = range(0, 3)

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


def train_arima_with_decomposition(data, save_dir="models"):
    """
    Melatih model ARIMA untuk setiap komponen dan menyimpan model ke file
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    models = {}
    for component in ["trend", "seasonal", "residual"]:
        best_params = find_best_parameters(data[component])
        print(f"Parameter terbaik untuk {component}: {best_params}")
        model = ARIMA(data[component], order=best_params)
        fitted_model = model.fit()
        models[component] = fitted_model

        # Save model to file
        model_filename = os.path.join(save_dir, f"arima_{component}_model.pkl")
        with open(model_filename, "wb") as file:
            pickle.dump(fitted_model, file)
        print(f"Model untuk {component} disimpan ke {model_filename}")

    return models


def plot_forecast(actual_data, forecast_df):
    """
    Memvisualisasikan data asli dan hasil prediksi 30 hari
    """
    plt.figure(figsize=(20, 10))

    # Plot data aktual
    plt.plot(
        actual_data.index, actual_data.values, label="Historical Data", color="blue"
    )

    # Plot prediksi
    plt.plot(
        forecast_df.index,
        forecast_df["forecast"],
        label="Forecast",
        color="red",
        linestyle="--",
    )

    # Plot interval kepercayaan
    plt.fill_between(
        forecast_df.index,
        forecast_df["lower_ci"],
        forecast_df["upper_ci"],
        color="red",
        alpha=0.1,
        label="95% Confidence Interval",
    )

    plt.title("30-Day Forecast with ARIMA (Decomposition)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Rotasi label tanggal
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Anotasi untuk nilai prediksi harian (mean per hari)
    daily_forecasts = forecast_df["forecast"].resample("D").mean()
    for date, value in daily_forecasts.items():
        plt.annotate(
            f"{value:.2f}",
            (date, value),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
        )

    plt.show()
    plt.savefig("forecast.png")


def clean_anomali_data(values, part_id):
    detail = get_detail(part_id=part_id)
    upper_threshold = detail[2]
    trip = (upper_threshold / 2 * 1.5) + upper_threshold

    print("upper threshold: ", upper_threshold)
    print("trip: ", trip)

    print("len values before cleansing: ", len(values))

    data = []

    for value in values:
        if value[1] > trip:
            continue
        else:
            data.append(value)

    print("len values after cleansing: ", len(data))
    return data


def main(part_id, features_id):
    # Mengambil data
    values = get_values(part_id, features_id)

    data = clean_anomali_data(values=values, part_id=part_id)
    df_decomposed = prepare_data_with_decomposition(data)
    models = train_arima_with_decomposition(df_decomposed)


if __name__ == "__main__":
    main("28516795-f22d-4cbc-9469-b720f5d881d7", "9dcb7e40-ada7-43eb-baf4-2ed584233de7")
