from database import get_connection
from datetime import datetime
from log import print_log
import pytz
import uuid


def get_parts():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT id, part_name, type_id FROM pf_parts WHERE web_id IS NOT NULL")
        tags = cur.fetchall()
        cur.close()
        conn.close()
        return tags
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred: {e}")
        
def get_non_dcs_parts():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT id, part_name, type_id FROM pf_parts WHERE  web_id IS NULL")
        tags = cur.fetchall()
        cur.close()
        conn.close()
        return tags
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred: {e}")   


def get_part(part_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT pf_parts.id, pf_parts.part_name, pf_parts.type_id, ms_equipment_master.name
            FROM pf_parts 
            JOIN ms_equipment_master ON pf_parts.equipment_id = ms_equipment_master.id
            WHERE pf_parts.id = %s
            """,
            (part_id,),
        )
        part = cur.fetchone()
        cur.close()
        conn.close()
        return part
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred: {e}")


def get_all_features():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name, category FROM dl_ms_features")
        features = cur.fetchall()
        cur.close()
        conn.close()
        return features
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred {e}")


def get_vibration_features(non_vibration_features):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, category FROM dl_ms_features WHERE id != %s",
            (non_vibration_features,),
        )
        features = cur.fetchall()
        cur.close()
        conn.close()
        return features
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred {e}")


def get_non_vibration_features(non_vibration_features):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, category FROM dl_ms_features WHERE id = %s",
            (non_vibration_features,),
        )
        features = cur.fetchall()
        cur.close()
        conn.close()
        return features
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred {e}")


def get_tags(*tag_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM dl_ms_tag WHERE id = %s", (tag_id[0],))

        tags = cur.fetchall()
        cur.close()
        conn.close()
        return tags
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred {e}")


def get_values(part_id, features_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT date_time, value
            FROM dl_features_data
            WHERE part_id = %s AND features_id = %s
            ORDER BY date_time ASC;
            """,
            (part_id, features_id),
        )

        # cur.execute(
        #     """
        #     SELECT date_time, value
        #     FROM dl_features_data
        #     WHERE part_id = %s
        #     AND date_time NOT BETWEEN '2024-11-01' AND '2024-11-06'
        #     ORDER BY date_time ASC;
        #     """,
        #     (part_id,),
        # )

        values = cur.fetchall()
        cur.close()
        conn.close()
        return values
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred {e}")


def save_predictions_to_db(forecast_df, part_id, features_id):
    """
    Menyimpan hasil prediksi ke database sesuai struktur tabel
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Hapus prediksi yang sudah ada
        delete_query = """
            DELETE FROM dl_predict 
            WHERE part_id = %s AND features_id = %s
        """

        # Insert data prediksi baru
        insert_query = """
            INSERT INTO dl_predict (
                id,
                features_id,
                pfi_value,
                status,
                date_time,
                part_id,
                created_at,
                updated_at
            ) VALUES (
                %s, %s, %s, 'normal', %s, %s, %s, %s
            )
            RETURNING id
        """

        # Reset index untuk mendapatkan timestamp sebagai kolom
        df_to_save = forecast_df.reset_index()
        df_to_save = df_to_save.rename(
            columns={"index": "date_time", "forecast": "value"}
        )

        # Hapus prediksi lama
        cur.execute(
            delete_query, (part_id, features_id)
        )

        # Insert prediksi baru
        inserted_count = 0
        for _, row in df_to_save.iterrows():
            predict_id = str(uuid.uuid4())
            now = datetime.now(pytz.timezone("Asia/Jakarta"))
            print("row datetime: ",row["date_time"])
            cur.execute(
                insert_query,
                (
                    predict_id,
                    row["features_id"],
                    row["value"],
                    row["date_time"],
                    row["part_id"],
                    now,
                    now,
                ),
            )
            inserted_count += 1

        conn.commit()
        print(f"Successfully saved {inserted_count} predictions to database")

    except Exception as e:
        print(f"Error saving predictions to database: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def get_predict_values(part_id):
    try:
        conn = get_connection()
        cur = conn.cursor()

        query = "SELECT id, pfi_value, date_time FROM dl_predict WHERE part_id = %s order by date_time asc"
        cur.execute(query, (part_id,))
        details = cur.fetchall()
        return details
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        if conn:
            conn.close()


def get_detail(part_id):
    try:
        conn = get_connection()
        cur = conn.cursor()

        query = "SELECT id, part_id, upper_threshold, lower_threshold, predict_status, time_failure, one_hundred_percent_condition FROM pf_details WHERE part_id = %s"
        cur.execute(query, (part_id,))
        details = cur.fetchone()
        return details
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        if conn:
            conn.close()


def update_detail(part_id, status, time_failure, predict_value,):
    try:
        conn = get_connection()
        cur = conn.cursor()
        now = datetime.now(pytz.timezone("Asia/Jakarta"))

        query = "UPDATE pf_details SET predict_status = %s, time_failure = %s, predict_value= %s, updated_at = %s WHERE part_id = %s"
        cur.execute(query, (status, time_failure,predict_value,now, part_id))
        conn.commit()
    except Exception as e:
        print(f"An exception occurred while updating: {e}")
    finally:
        if conn:
            conn.close()

def update_percent_condition(part_id, percent_condition, warning_percent_condition):
    try:
        conn = get_connection()
        cur = conn.cursor()
        now = datetime.now(pytz.timezone("Asia/Jakarta"))


        query = "UPDATE pf_details SET percent_condition = %s, warning_percent_condition = %s, updated_at = %s WHERE part_id = %s"
        cur.execute(query, (percent_condition, warning_percent_condition, now, part_id))
        conn.commit()
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        if conn:
            conn.close()

def get_current_feature_value(part_id, feature_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Query untuk mengambil data
        query = """
            SELECT * FROM dl_features_data WHERE part_id = %s AND features_id = %s
            ORDER BY date_time DESC LIMIT 1
        """

        cursor.execute(query, (part_id, feature_id))

        # Mendapatkan nama kolom
        columns = [col[0] for col in cursor.description]

        # Mendapatkan hasil dari query
        data = cursor.fetchone()

        cursor.close()
        conn.close()

        # Mengonversi setiap tuple menjadi dictionary
        return data[3]
    except Exception as e:
        raise Exception(f"Error: {e}")
    

def save_process_logs(process_type, level, message, detail=None):
    try:
        conn = get_connection()
        cur = conn.cursor()

        now = datetime.now(pytz.timezone("Asia/Jakarta"))

        query = "INSERT INTO pf_process_logs (id, process_type, timestamp, level, message, detail, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        cur.execute(query, (str(uuid.uuid4()), process_type, now, level, message,detail, now, now))
        conn.commit()
        
    except Exception as e:
        print(f"An exception occurred while saving log: {e}")
    finally:
        if conn:
            conn.close()

def create_process_monitoring(
    process_type,
    start_time,
    end_time,
    total_sensor,
    data_row_count,
    data_size_mb,
    disk_usage_percentage,
    sensor_data_percentage,
    status,
    ):
    try:
        conn = get_connection()

        cur = conn.cursor()
        id = str(uuid.uuid4())
        now = datetime.now(pytz.timezone("Asia/Jakarta"))

        query = """
            INSERT INTO pf_process_monitoring (
                id,
                process_type,
                start_timestamp,
                finish_timestamp,
                total_data,
                data_row_count,
                data_size_mb,
                disk_usage_percentage,
                sensor_data_percentage,
                status,
                created_at,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cur.execute(query, (
            id,
            process_type,
            start_time,
            end_time,
            total_sensor,
            data_row_count,
            data_size_mb,
            disk_usage_percentage,
            sensor_data_percentage,
            status,
            now,
            now
        ))
        conn.commit()
        
        return {
            "process_monitoring_id": id,
        }
    except Exception as e:
        print(f"An exception occurred while creating monitoring: {e}")
    finally:
        if conn:
            conn.close()
            
def update_process_monitoring(
    process_monitoring_id,
    total_sensor,
    end_time,
    data_row_count,
    data_size_mb,
    disk_usage_percentage,
    sensor_data_percentage,
    status,
    ):
    try:
        conn = get_connection()

        cur = conn.cursor()
        now = datetime.now(pytz.timezone("Asia/Jakarta"))

        query = """
            UPDATE pf_process_monitoring
            SET total_data = %s,
                finish_timestamp = %s,
                data_row_count = %s,
                data_size_mb = %s,
                disk_usage_percentage = %s,
                sensor_data_percentage = %s,
                status = %s,
                updated_at = %s
            WHERE id = %s
        """

        cur.execute(query, (
            total_sensor,
            end_time,
            data_row_count,
            data_size_mb,
            disk_usage_percentage,
            sensor_data_percentage,
            status,
            now,
            process_monitoring_id
        ))
        conn.commit()
        
    except Exception as e:
        print(f"An exception occurred while updating monitoring: {e}")
    finally:
        if conn:
            conn.close()
def get_process_monitoring(process_monitoring_id):
    try:
        conn = get_connection()
        cur = conn.cursor()

        query = """
            SELECT 
                ppm.*
            FROM pf_process_monitoring ppm
            WHERE ppm.id = %s
        """

        cur.execute(query, (process_monitoring_id,))
        process_monitoring = cur.fetchone()
        return dict(zip([col[0] for col in cur.description], process_monitoring))
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        if conn:
            conn.close()

def update_total_data_and_data_row(process_monitoring_id, total_data, data_row_count):
    try:
        conn = get_connection()
        cur = conn.cursor()
        now = datetime.now(pytz.timezone("Asia/Jakarta"))

        query = """
            UPDATE pf_process_monitoring
            SET total_data = %s,
                data_row_count = %s,
                updated_at = %s
            WHERE id = %s
        """

        cur.execute(query, (total_data, data_row_count,now, process_monitoring_id))
        conn.commit()
    except Exception as e:
        print(f"An exception occurred while updating monitoring: {e}")
    finally:
        if conn:
            conn.close()
            
def disk_usage_count():
    try:
        conn = get_connection()
        cur = conn.cursor()

        query = """
        SELECT 
            ROUND(pg_total_relation_size('dl_predict') / 1048576.0, 2) AS table_size_mb,
            ROUND(1267110555648 / 1048576.0, 2) AS total_storage_mb,
            ROUND((pg_total_relation_size('dl_predict') * 100.0 / 1267110555648), 4) AS usage_percent;
        """
        
        cur.execute(query)
        disk_usage = cur.fetchone()
        return disk_usage
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        if conn:
            conn.close()
    