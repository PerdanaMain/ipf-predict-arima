import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pytz
import concurrent.futures
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from model import get_values, get_all_tags
from sklearn.metrics import mean_squared_error  # type: ignore
from math import sqrt
from datetime import datetime
from joblib import Parallel, delayed  # type: ignore


def find_best_arima(data, p_range, d_range, q_range):
    """Grid search untuk menemukan parameter ARIMA terbaik berdasarkan AIC dengan paralelisasi."""
    best_aic, best_order = float("inf"), None

    def evaluate_arima(order):
        try:
            model = ARIMA(data, order=order)
            model_fit = model.fit()
            return model_fit.aic, order
        except:
            return float("inf"), None

    # Paralelisasi evaluasi ARIMA untuk setiap kombinasi parameter
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_arima)(order)
        for order in itertools.product(p_range, d_range, q_range)
    )

    # Pilih parameter dengan AIC terbaik
    for aic, order in results:
        if aic < best_aic:
            best_aic, best_order = aic, order

    return best_order


def train_arima_model(train_data, order):
    """Melatih model ARIMA dengan parameter yang diberikan."""
    model = ARIMA(train_data, order=order)
    return model.fit()


def plot_results(test, predictions, title="Actual vs Predicted"):
    """Menampilkan plot data aktual dan prediksi."""
    plt.figure(figsize=(10, 5))
    plt.plot(test, label="Actual")
    plt.plot(predictions, color="red", label="Predicted")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_future_forecast(data, future_forecast, n_steps):
    """Menampilkan plot data aktual dengan prediksi masa depan."""
    days = n_steps // 1440  # Konversi menit ke hari
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Historical Data")
    future_index = np.arange(len(data), len(data) + n_steps)
    plt.plot(future_index, future_forecast, color="green", label="3-Month Forecast")
    plt.title(f"Forecast for Next {days} Days")
    plt.legend()
    plt.show()


def main(tag_id):
    # Mengambil data dari database
    data = get_values(tag_id)
    if data is None:
        print("Failed to fetch data from database.")
        return Exception("Failed to fetch data from database.")

    X = np.array([item[2] for item in data])

    # Grid search parameter ARIMA terbaik
    best_order = find_best_arima(
        X, p_range=range(0, 4), d_range=range(0, 2), q_range=range(0, 4)
    )
    print(f"Best ARIMA order: {best_order}")

    # Membagi data menjadi pelatihan dan pengujian
    split_index = int(len(X) * 0.66)
    train, test = X[:split_index], X[split_index:]

    # Melatih model dengan parameter terbaik
    model_fit = train_arima_model(train, best_order)

    # Memprediksi data pengujian
    predictions = model_fit.forecast(steps=len(test))
    rmse = sqrt(mean_squared_error(test, predictions))
    print(f"Test RMSE: {rmse:.3f}")

    # Menampilkan plot perbandingan aktual dan prediksi
    # plot_results(test, predictions)

    # Prediksi 7 hari
    n_minutes = 10080
    future_forecast = model_fit.forecast(steps=n_minutes)
    print(f"Forecast for next {n_minutes // 1440} days: {future_forecast}")

    # Plot hasil prediksi masa depan
    # plot_future_forecast(X, future_forecast, n_minutes)

    # Plot perbedaan antara data aktual dan prediksi
    difference(X, future_forecast, n_minutes)


def difference(X, future_forecast, n_minutes):
    days = n_minutes // 1440

    # Ambil data yang tidak sama/hilangkan grafik lurus dari future_forecast
    selected_data = []
    for i in range(len(future_forecast) - 1):
        if round(future_forecast[i], 5) != round(future_forecast[i + 1], 5):
            selected_data.append(future_forecast[i])

    selected_actual_data = X[-len(selected_data) :]

    print(len(selected_data), len(selected_actual_data))

    # Plotting data
    # plt.figure(figsize=(12, 6))
    # plt.plot(X, label="Historical Data")
    # future_index = np.arange(len(X), len(X) + len(selected_data))
    # plt.plot(future_index, selected_data, color="green", label="Selected Forecast")
    # plt.title(f"Forecast for Next {days} Days")
    # plt.legend()
    # plt.show()


def index():
    time_start = datetime.now(pytz.timezone("Asia/Jakarta")).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    print(f"Started training at: {time_start}")

    tags = get_all_tags()

    # Menggunakan ThreadPoolExecutor atau ProcessPoolExecutor untuk paralelisasi
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Menjalankan fungsi main secara paralel untuk setiap tag
        futures = [executor.submit(main, tag[0]) for tag in tags]

        # Menunggu semua thread selesai dan menangani error jika ada
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Dapatkan hasil untuk memeriksa apakah ada error
            except Exception as e:
                print(f"Error in thread: {e}")

    time_end = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started at: {time_start}, Ended at: {time_end}")


if __name__ == "__main__":
    try:
        index()  # Menjalankan fungsi utama
    except KeyboardInterrupt:
        print("Program stopped by user.")
        sys.exit()
