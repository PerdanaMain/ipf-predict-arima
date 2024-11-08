import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from model import get_values


def index():
    print("========== Get Data From Database ==========")
    data = get_values(1)
    data = np.array([item[2] for item in data])

    # Plot original data
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Original Data")
    plt.title("Original Data Plot")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    # Split data into train and test sets
    split_index = len(data) // 2
    train = data[:split_index]
    test = data[split_index:]

    # Fit ARIMA model (p, d, q)
    model = ARIMA(
        train, order=(5, 1, 0)
    )  # p=5, d=1, q=0; sesuaikan parameter sesuai data Anda
    model_fit = model.fit()

    # Predict on the test set
    pred_test = model_fit.forecast(steps=len(test))

    # Plot actual vs predicted data
    plt.figure(figsize=(12, 6))
    plt.plot(
        range(len(train), len(train) + len(test)),
        test,
        label="Actual Test Data",
        color="b",
    )
    plt.plot(
        range(len(train), len(train) + len(test)),
        pred_test,
        label="Predicted Test Data",
        color="r",
        linestyle="--",
    )
    plt.title("Test Data: Actual vs Predicted using ARIMA")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    # Plot model summary (optional)
    print(model_fit.summary())


if __name__ == "__main__":
    index()
