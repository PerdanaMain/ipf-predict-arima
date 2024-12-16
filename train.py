from datetime import datetime, timedelta
from model import *
import pandas as pd  # type: ignore
import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from statsmodels.tsa.stattools import adfuller  # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error  # type: ignore
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


def find_best_parameters(data):
    """
    Mencari parameter terbaik untuk model ARIMA
    """
    best_aic = float("inf")
    best_params = None

    p_values = range(0, 3)
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

    return best_params


def train_arima_model(data, order):
    """
    Melatih model ARIMA dengan parameter yang ditentukan
    """
    model = ARIMA(data, order=order)
    return model.fit()


def make_weekly_forecast(model, start_date):
    """
    Membuat prediksi untuk 7 hari ke depan (168 jam)
    """
    # Membuat prediksi dengan interval kepercayaan
    forecast_result = model.get_forecast(steps=168)

    # Mendapatkan prediksi mean dan interval kepercayaan
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # Membuat index datetime untuk 7 hari ke depan
    forecast_index = pd.date_range(start=start_date, periods=168, freq="H")

    # Membuat DataFrame dengan prediksi dan interval kepercayaan
    forecast_df = pd.DataFrame(
        {
            "forecast": forecast_mean,
            "lower_ci": conf_int.iloc[:, 0],
            "upper_ci": conf_int.iloc[:, 1],
        },
        index=forecast_index,
    )

    return forecast_df


def plot_data_preparation(values):
    """
    Memvisualisasikan data sebelum dan sesudah prepare_data()
    """
    # Data original
    df_original = pd.DataFrame(values, columns=["timestamp", "value"])
    df_original.set_index("timestamp", inplace=True)
    df_original.sort_index(inplace=True)

    # Data setelah preparation
    df_prepared = prepare_data(values)

    # Membuat plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot data original
    ax1.plot(
        df_original.index,
        df_original["value"],
        "o-",
        label="Data Original",
        color="blue",
        markersize=4,
    )
    ax1.set_title("Data Original")
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Value")
    ax1.grid(True)
    ax1.tick_params(axis="x", rotation=45)

    # Plot data setelah preparation
    ax2.plot(
        df_prepared.index,
        df_prepared["value"],
        "o-",
        label="Data Setelah Preparation",
        color="red",
        markersize=2,
    )
    ax2.set_title("Data Setelah Preparation (Resampling & Interpolasi)")
    ax2.set_xlabel("Timestamp")
    ax2.set_ylabel("Value")
    ax2.grid(True)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    # Print informasi tambahan
    print("\nInformasi Data:")
    print(f"Jumlah data original: {len(df_original)}")
    print(f"Jumlah data setelah preparation: {len(df_prepared)}")
    print(f"\nRange waktu data:")
    print(f"Start: {df_original.index.min()}")
    print(f"End: {df_original.index.max()}")

    # Hitung gap dalam data original
    time_diff = df_original.index.to_series().diff()
    gaps = time_diff[time_diff > pd.Timedelta(hours=1)]
    if not gaps.empty:
        print("\nGap dalam data original (> 1 jam):")
        for idx, gap in gaps.items():
            print(f"Gap pada {idx}: {gap}")


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
    plt.title("7-Day Forecast with ARIMA")
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


def main():
    # Mengambil data
    values = get_values("386fe843-5e4e-4647-8623-3eaa4d4eba28")
    # plot_data_preparation(values)

    # Menyiapkan data
    df = prepare_data(values)

    # Mencari parameter terbaik
    best_params = find_best_parameters(df["value"])
    print(f"Parameter ARIMA terbaik: {best_params}")

    # Melatih model dengan seluruh data
    model = train_arima_model(df["value"], order=best_params)

    # Membuat prediksi 7 hari ke depan
    last_date = df.index[-1]
    forecast = make_weekly_forecast(model, last_date)

    print(forecast)

    # Menampilkan visualisasi
    # plot_forecast(df["value"], forecast)


if __name__ == "__main__":
    main()
