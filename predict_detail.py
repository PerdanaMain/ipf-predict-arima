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
    upper_threshold = detail[2]
    one_percent_condition = detail[6]
    
    current_value = get_current_feature_value(part_id, feature_id=feature_id)

    # Hitung percent_condition
    percent_condition = abs(upper_threshold - current_value) / abs(upper_threshold - one_percent_condition) * 100
    
    # Interpolasi jika melebihi 100%
    percent_condition = round(percent_condition, 2)
    
    update_percent_condition(part_id, percent_condition)
    

def main(part_id):
    print("mengambil data ...")
    detail = get_detail(part_id)
    predict_values = get_predict_values(part_id)
    features_id = "9dcb7e40-ada7-43eb-baf4-2ed584233de7"

    print("menghitung status ...")
    result = checking_status(predict_values, detail)

    # select predicted failed
    predicted_failed = [item for item in result if item["status"] == "predicted failed"]
    predicted_warning = [item for item in result if item["status"] == "warning"]
    
    # select once
    predicted_failed = [predicted_failed[0]] if predicted_failed else []
    predicted_warning = [predicted_warning[0]] if predicted_warning else []
    
    print(result)
    
    if len(predicted_failed) != 0:
        update_detail(part_id, "predicted failed", predicted_failed[0]["datetime"], predicted_failed[0]["value"])
    elif len(predicted_warning) != 0:
        update_detail(part_id, "warning", predicted_warning[0]["datetime"], predicted_warning[0]["value"])
    else:
        update_detail(part_id, "normal", None, None)    
    
    # print("menghitung persentase kondisi ...")
    percent_calculation(part_id, features_id)

    print("done")


if __name__ == "__main__":
    # main()
    main("0d25b4e6-972b-4c40-b964-d17e09a3d3fa")
    # percent_calculation("64492e3f-8e1f-4eb4-b9ea-8a2ead652c8e", "9dcb7e40-ada7-43eb-baf4-2ed584233de7")
    # print("test command")
