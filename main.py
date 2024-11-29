from matplotlib import pyplot as plt # type: ignore
from requests import get # type: ignore
from model import *
from train import find_best_arima, train_arima_model
from datetime import timedelta
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from log import print_log
import numpy as np # type: ignore
import pytz
import time

def execute_arima(equipment_id, features_id):
    data = get_values(equipment_id, features_id)

    if len(data) == 0:
        print(f"No data found for equipment_id: {equipment_id}, features_id: {features_id}")
        return

    # Ekstrak value dan timestamp
    raw_values = [val[3] for val in data] 
    timestamps = [val[2] for val in data] 

    X = []
    for value in raw_values:
        try:
            X.append(float(value))
        except (ValueError, TypeError):
            # Jika konversi gagal, masukkan nilai default (0) atau nilai lainnya
            X.append(0.0)

    if len(X) < 10:  # Minimum data untuk ARIMA
        print(f"Insufficient data for ARIMA: {len(X)} records found.")
        return

    # Gunakan 50% data awal untuk pencarian parameter ARIMA
    subset = X[: int(len(X) * 0.5)]
    best_order = find_best_arima(
        subset, p_range=range(0, 3), d_range=range(0, 2), q_range=range(0, 3)
    )
    print(f"Best ARIMA order for equipment_id {equipment_id}, features_id {features_id}: {best_order}")

    # Membagi data menjadi pelatihan dan pengujian
    split_index = int(len(X) * 0.66)
    train, test = X[:split_index], X[split_index:]

    model_fit = train_arima_model(train, best_order)

    n_days = 7
    future_forecast = model_fit.forecast(steps=n_days)

    # Membuat timestamp untuk 7 hari ke depan
    last_date = timestamps[-1]
    future_timestamps = [(last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d") for i in range(n_days)]

    # delete old prediction
    delete_predicts(equipment_id, features_id)

    # Simpan hasil prediksi
    create_predict(equipment_id, features_id, future_forecast, future_timestamps)

    print(f"ARIMA prediction for equipment_id: {equipment_id} finished.")



def index():
    features = get_all_features() 
    equipments = get_all_equipment()  
    time = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
    print_log(f"Starting ARIMA prediction at {time}")
    print(f"Starting ARIMA prediction at {time}")

    def process(equipment, feature):
        """Fungsi pembantu untuk dieksekusi secara paralel."""
        try:
            execute_arima(equipment, feature)  
        except Exception as e:
            print(f"Error processing tag {equipment} and feature {feature}: {e}")
            print_log(f"Error processing tag {equipment} and feature {feature}: {e}")

    with ThreadPoolExecutor() as executor:
        try:
            futures = [
                executor.submit(process, equipment[0], feature[0])
                for equipment in equipments
                for feature in features
            ]
            for future in futures:
                future.result()  
        except Exception as e:
            print(f"An exception occurred: {e}")
            print_log(f"An exception occurred: {e}")

    time = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
    print_log(f"ARIMA prediction finished at {time}")
    print(f"ARIMA prediction finished at {time}")



if __name__ == "__main__":
    # index()        

    while True:
        date = datetime.now(pytz.timezone("Asia/Jakarta"))

        index()        
        
        next_execution = (datetime.now(pytz.timezone("Asia/Jakarta")).replace(hour=5, minute=0, second=0, microsecond=0) + timedelta(days=1))
        wait_time = (next_execution - datetime.now(pytz.timezone("Asia/Jakarta"))).total_seconds()

        print_log(f"Next execution scheduled at: {next_execution}")
        print("Next execution scheduled at: ", next_execution)

        time.sleep(wait_time)

