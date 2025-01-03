from model import get_detail, get_predict_values, update_detail


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

        if predict_value > upper_threshold or predict_value < lower_threshold:
            result.append(
                {"datetime": dt, "status": "predicted failed", "value": predict_value}
            )
        else:
            result.append({"datetime": dt, "status": "normal", "value": predict_value})

    return result


def main():
    part_id = "4909d836-c26c-4f39-ba64-7cc9cff5e400"
    print("mengambil data ...")
    detail = get_detail(part_id)
    predict_values = get_predict_values(part_id)

    print("menghitung status ...")
    result = checking_status(predict_values, detail)

    # select predicted failed
    predicted_failed = [item for item in result if item["status"] == "predicted failed"]
    if predicted_failed:
        predicted_failed = [predicted_failed[0]]
        update_detail(
            part_id, predicted_failed[0]["status"], predicted_failed[0]["datetime"]
        )
    else:
        update_detail(part_id, "normal", None)

    print("done")


if __name__ == "__main__":
    main()
