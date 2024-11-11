import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from model import get_values
from sklearn.metrics import mean_squared_error  # type: ignore
from math import sqrt
import itertools


def find_best_arima(data, p_range, d_range, q_range):
    """Grid search untuk menemukan parameter ARIMA terbaik berdasarkan AIC."""
    best_aic, best_order = float("inf"), None
    for order in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(data, order=order)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic, best_order = model_fit.aic, order
        except:
            continue
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


def main():
    # Mengambil data dari database
    data = get_values(1)

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
    plot_results(test, predictions)

    # Prediksi 3 bulan ke depan (asumsinya data harian; 3 bulan = 90 hari)
    n_minutes = 129600
    future_forecast = model_fit.forecast(steps=n_minutes)
    print(f"Forecast for next {n_minutes} days: {future_forecast}")

    # Plot hasil prediksi masa depan
    plot_future_forecast(X, future_forecast, n_minutes)


if __name__ == "__main__":
    main()
