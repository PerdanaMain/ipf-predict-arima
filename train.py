from datetime import datetime, timedelta
from model import *
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


def prepare_data(values):
    """
    Menyiapkan data time series dalam format yang sesuai
    """
    df = pd.DataFrame(values, columns=["timestamp", "value"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df = df.resample("h").mean()
    df = df.interpolate(method="linear")
    return df


def decompose_timeseries(data):
    """
    Melakukan dekomposisi time series menjadi komponen trend, seasonal, dan residual
    """
    # Lakukan dekomposisi
    decomposition = seasonal_decompose(data, period=24)  # period=24 untuk data per jam

    # Dapatkan komponen-komponennya
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot hasil dekomposisi
    # plt.figure(figsize=(15, 12))
    # plt.subplot(411)
    # plt.plot(data)
    # plt.title("Original Data")
    # plt.subplot(412)
    # plt.plot(trend)
    # plt.title("Trend")
    # plt.subplot(413)
    # plt.plot(seasonal)
    # plt.title("Seasonal")
    # plt.subplot(414)
    # plt.plot(residual)
    # plt.title("Residual")
    # plt.tight_layout()
    # plt.show()

    return trend, seasonal, residual


def prepare_data_with_decomposition(values):
    """
    Menyiapkan data dengan dekomposisi
    """
    # Persiapan data awal
    df = prepare_data(values)

    # Lakukan dekomposisi
    trend, seasonal, residual = decompose_timeseries(df["value"])

    # Hapus NaN values yang muncul dari dekomposisi
    valid_idx = ~(trend.isna() | seasonal.isna() | residual.isna())
    trend = trend[valid_idx]
    seasonal = seasonal[valid_idx]
    residual = residual[valid_idx]

    # Simpan komponen dalam DataFrame
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

    # Perluas range pencarian parameter
    p_values = range(0, 5)
    d_values = range(0, 3)
    q_values = range(0, 5)

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

    return best_params


def train_arima_with_decomposition(data):
    """
    Melatih model ARIMA untuk setiap komponen
    """
    models = {}
    forecasts = {}

    for component in ["trend", "seasonal", "residual"]:
        # Cari parameter terbaik untuk setiap komponen
        best_params = find_best_parameters(data[component])
        print(f"Parameter terbaik untuk {component}: {best_params}")

        # Train model untuk setiap komponen
        model = ARIMA(data[component], order=best_params)
        fitted_model = model.fit()

        models[component] = fitted_model

    return models


def forecast_with_decomposition(models, days=7):
    """
    Membuat prediksi dengan menggabungkan hasil prediksi setiap komponen
    Parameter:
    days (int): Jumlah hari yang akan diprediksi
    """
    # Konversi hari ke steps sesuai tipe data
    # Cek frekuensi dari index data asli
    sample_model = list(models.values())[0]
    if isinstance(sample_model.data, pd.Series):
        freq = sample_model.data.index.freq
    else:
        freq = sample_model.data.dates.freq

    # Tentukan steps berdasarkan frekuensi
    if freq is None:
        # Default ke daily jika freq tidak terdeteksi
        steps = days
    elif isinstance(freq, pd.offsets.Hour):
        steps = days * 24
    else:
        steps = days

    forecasts = {}
    # Prediksi untuk setiap komponen
    for component, model in models.items():
        forecast = model.forecast(steps=steps)
        forecasts[component] = forecast

    # Gabungkan prediksi
    final_forecast = forecasts["trend"] + forecasts["seasonal"] + forecasts["residual"]

    return final_forecast


def plot_forecast(actual_data, forecast_df):
    """
    Memvisualisasikan data asli dan hasil prediksi
    """
    plt.figure(figsize=(15, 8))

    # Plot data aktual
    plt.plot(
        actual_data.index,
        actual_data.values,
        label="Historical Data",
        color="blue",
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

    # Menambahkan label dan judul
    plt.title(f"{len(forecast_df)/24:.0f}-Day Forecast with ARIMA (Decomposition)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Rotasi label tanggal
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Menambahkan anotasi untuk nilai prediksi harian
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


def prepare_vib_data(values):
    """
    Menyiapkan data time series untuk sensor vibrasi dalam format yang sesuai
    """
    df = pd.DataFrame(values, columns=["timestamp", "value"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    # Resampling ke data harian untuk prediksi per hari
    df = df.resample("D").mean()
    df = df.interpolate(method="linear")
    return df


def decompose_vib_timeseries(data):
    """
    Melakukan dekomposisi time series untuk data vibrasi
    """
    # Dekomposisi dengan periode 7 untuk pola mingguan
    decomposition = seasonal_decompose(data, period=7)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    return trend, seasonal, residual


def prepare_vib_data_with_decomposition(values):
    """
    Menyiapkan data vibrasi dengan dekomposisi
    """
    df = prepare_vib_data(values)

    trend, seasonal, residual = decompose_vib_timeseries(df["value"])

    valid_idx = ~(trend.isna() | seasonal.isna() | residual.isna())
    trend = trend[valid_idx]
    seasonal = seasonal[valid_idx]
    residual = residual[valid_idx]

    df_decomposed = pd.DataFrame(
        {"trend": trend, "seasonal": seasonal, "residual": residual}
    )

    return df_decomposed


def plot_vib_forecast(actual_data, forecast_df):
    """
    Memvisualisasikan data asli dan hasil prediksi untuk sensor vibrasi
    """
    plt.figure(figsize=(15, 8))

    plt.plot(
        actual_data.index, actual_data.values, label="Historical Data", color="blue"
    )

    plt.plot(
        forecast_df.index,
        forecast_df["forecast"],
        label="Forecast",
        color="red",
        linestyle="--",
    )

    plt.fill_between(
        forecast_df.index,
        forecast_df["lower_ci"],
        forecast_df["upper_ci"],
        color="red",
        alpha=0.1,
        label="95% Confidence Interval",
    )

    plt.title(f"{len(forecast_df)}-Day Daily Forecast with ARIMA (Vibration)")
    plt.xlabel("Date")
    plt.ylabel("Vibration Value")
    plt.legend()
    plt.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Anotasi untuk nilai prediksi
    for date, value in forecast_df["forecast"].items():
        plt.annotate(
            f"{value:.2f}",
            (date, value),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
        )

    plt.show()


def start_vib_train(part_id, prediction_days=7):
    """
    Fungsi untuk melatih dan memprediksi data sensor vibrasi
    Parameter:
    part_id: ID komponen
    prediction_days (int): Jumlah hari yang akan diprediksi
    """
    # Mengambil data
    values = get_values(part_id)

    # Siapkan data dengan dekomposisi
    df_decomposed = prepare_vib_data_with_decomposition(values)

    # Train model untuk setiap komponen
    models = train_arima_with_decomposition(df_decomposed)

    # Buat prediksi
    forecast = forecast_with_decomposition(models, days=prediction_days)

    # Buat DataFrame untuk hasil prediksi
    last_date = df_decomposed.index[-1]
    forecast_index = pd.date_range(
        start=last_date, periods=prediction_days + 1, freq="D"
    )[1:]

    forecast_df = pd.DataFrame(
        {
            "forecast": forecast,
            "lower_ci": forecast - 2 * forecast.std(),
            "upper_ci": forecast + 2 * forecast.std(),
        },
        index=forecast_index,
    )

    # Plot hasil
    # plot_vib_forecast(df_decomposed.sum(axis=1), forecast_df)

    return forecast_df


def start_non_vib_train(part_id, prediction_days=7):
    """
    Fungsi untuk melatih dan memprediksi data sensor non-vibrasi
    Parameter:
    part_id: ID komponen
    prediction_days (int): Jumlah hari yang akan diprediksi
    """
    # Mengambil data
    values = get_values(part_id)

    # Siapkan data dengan dekomposisi
    df_decomposed = prepare_data_with_decomposition(values)

    # Train model untuk setiap komponen
    models = train_arima_with_decomposition(df_decomposed)

    # Buat prediksi
    forecast = forecast_with_decomposition(models, days=prediction_days)

    # Buat DataFrame untuk hasil prediksi
    last_date = df_decomposed.index[-1]
    forecast_index = pd.date_range(
        start=last_date, periods=(prediction_days * 24) + 1, freq="H"
    )[1:]

    forecast_df = pd.DataFrame(
        {
            "forecast": forecast,
            "lower_ci": forecast - 2 * forecast.std(),
            "upper_ci": forecast + 2 * forecast.std(),
        },
        index=forecast_index,
    )

    # Plot hasil
    # plot_forecast(df_decomposed.sum(axis=1), forecast_df)

    return forecast_df


def main():
    # Tentukan jumlah hari prediksi
    prediction_days = 30  # Bisa diubah sesuai kebutuhan

    tag = get_part("cceddfc8-1556-49f9-8727-894a3e0551ff")

    if tag[0][2] == "b45a04c6-e2e2-465a-ad84-ccefe0f324d2":  # id vibration
        start_vib_train(tag[0][0], prediction_days)
    else:
        start_non_vib_train(tag[0][0], prediction_days)


if __name__ == "__main__":
    main()
