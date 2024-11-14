from matplotlib import pyplot as plt
from model import get_values, create_predict
from train import find_best_arima, train_arima_model
from sklearn.metrics import mean_squared_error  # type: ignore
from math import sqrt
from datetime import timedelta
import numpy as np


def index():
    tag_id = 1
    data = get_values(tag_id)

    if len(data) == 0:
        print("No data found")
        return

    # Ekstrak value dan timestamp
    X = np.array([val[2] for val in data])
    timestamps = [val[1] for val in data]

    subset = X[: int(len(X) * 0.5)]  # Gunakan 50% data awal untuk pencarian cepat

    # Grid search parameter ARIMA terbaik
    best_order = find_best_arima(
        subset, p_range=range(0, 3), d_range=range(0, 2), q_range=range(0, 3)
    )
    print(f"Best ARIMA order: {best_order}")

    # Membagi data menjadi pelatihan dan pengujian
    split_index = int(len(X) * 0.66)
    train, test = X[:split_index], X[split_index:]
    train_timestamps, test_timestamps = (
        timestamps[:split_index],
        timestamps[split_index:],
    )

    # Melatih model dengan parameter terbaik
    model_fit = train_arima_model(train, best_order)

    n_minutes = 1440 * 7  # Prediksi 7 hari ke depan
    future_forecast = model_fit.forecast(steps=n_minutes)

    future_timestamps = [
        timestamps[-1] + timedelta(minutes=i) for i in range(1, n_minutes + 1)
    ]

    create_predict(tag_id, future_forecast, future_timestamps)

    print("ARIMA prediction finished.")


def plot_future_forecast(data, future_forecast, n_steps, timestamps):
    """Menampilkan plot data aktual dengan prediksi masa depan."""
    days = n_steps // 1440  # Konversi menit ke hari

    # Buat timestamp untuk 7 hari ke depan dengan interval per menit
    future_timestamps = [
        timestamps[-1] + timedelta(minutes=i) for i in range(1, n_steps + 1)
    ]

    print("Length of timestamps: ", len(timestamps))
    print("Length of data: ", len(data))
    print("Length of future forecast: ", len(future_forecast))
    print("Length of future timestamps: ", len(future_timestamps))

    print("Length of timestamps: ", len(timestamps))
    print(timestamps)

    # # Cetak beberapa contoh timestamp untuk memastikan hasil
    # print("Contoh timestamp masa depan:", future_timestamps[:10])

    # plt.figure(figsize=(12, 6))
    # # plt.plot(timestamps, data, label="Historical Data", color="blue")
    # plt.plot(future_timestamps, future_forecast, color="green", label="7-Day Forecast")
    # plt.xlabel("Timestamp")
    # plt.ylabel("Values")
    # plt.title(f"Forecast for Next {days} Days")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    index()
