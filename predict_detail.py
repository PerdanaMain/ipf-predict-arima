from model import get_current_feature_value, get_detail, get_predict_values, update_detail, update_percent_condition
from datetime import datetime, timedelta


def checking_status(predict_values, detail):
    """
    Mencari status prediksi berdasarkan upper dan lower threshold
    """
    upper_threshold = detail[2]
    lower_threshold = detail[3]

    result = []
    
    for value in predict_values:
        predict_value = value[1]  # nilai prediksi ada di index 1
        dt = value[2]  # datetime ada di index 2
        # print(value[1] < lower_threshold)

        if predict_value >= lower_threshold and predict_value <= upper_threshold:
            result.append({"datetime": dt, "status": "warning", "value": predict_value})
        elif predict_value > upper_threshold:
            result.append(
                {"datetime": dt, "status": "predicted failed", "value": predict_value }
            )
        else:
            result.append({"datetime": dt, "status": "normal", "value": predict_value})

    return result

def percent_calculation(part_id, feature_id):
    detail = get_detail(part_id)
    upper_threshold = detail[2]  # fail threshold
    lower_threshold = detail[3]  # warning threshold
    one_percent_condition = detail[6]  # normal value
    
    current_value = get_current_feature_value(part_id, feature_id=feature_id)

    # Hitung percent_condition menggunakan batas fail
    percent_condition = abs(upper_threshold - current_value) / abs(upper_threshold - one_percent_condition) * 100
    
    # Set warning_percent sama dengan percent_condition jika current_value dalam range normal
    warning_percent = percent_condition if current_value <= lower_threshold else 100
    
    percent_condition = round(percent_condition, 2)
    warning_percent = round(warning_percent, 2)
    
    update_percent_condition(part_id, percent_condition, warning_percent)
    

def main(part_id):
    print("mengambil data ...")
    detail = get_detail(part_id)
    predict_values = get_predict_values(part_id)
    features_id = "9dcb7e40-ada7-43eb-baf4-2ed584233de7"

    print("menghitung status ...")
    result = checking_status(predict_values, detail)
    
    # select predicted failed
    predicted_failed = [item for item in result if item["status"] == "predicted failed"]
    
    # select once
    # predicted_failed = [predicted_failed[0]] if predicted_failed is not None else []
    
    print(len(predicted_failed) != 0)

    if len(predicted_failed) != 0:
        update_detail(part_id, "predicted failed", predicted_failed[0]["datetime"], predicted_failed[0]["value"], f"Terdeteksi failure sampai waktu {predicted_failed[0]['datetime']}")
    

if __name__ == "__main__":
    main("e1d5179b-f7c9-449d-ad49-047d13fb5acc")
    # percent_calculation("64492e3f-8e1f-4eb4-b9ea-8a2ead652c8e", "9dcb7e40-ada7-43eb-baf4-2ed584233de7")
    # print("test command")
