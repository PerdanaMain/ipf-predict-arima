from matplotlib import pyplot as plt
from model import get_values
from non_vibration_train import execute_arima


def index():
    data = get_values(1)

    values = [val[2] for val in data]
    timestamps = [val[1] for val in data]

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, values)
    plt.title("Historical Data")
    plt.show()


if __name__ == "__main__":
    index()
