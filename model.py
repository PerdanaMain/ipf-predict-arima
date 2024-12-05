from database import get_connection
from datetime import datetime
from log import print_log
import pytz
import uuid


def get_all_equipment():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT id, part_name FROM pf_parts")
        tags = cur.fetchall()
        cur.close()
        conn.close()
        return tags
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


def get_values(part_id,features_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, part_id, date_time, value 
            FROM dl_features_data
            WHERE part_id = %s AND features_id = %s
            """,
            (part_id, features_id),
        )
        values = cur.fetchall()
        cur.close()
        conn.close()
        print("Data fetched successfully, count: ", len(values))
        return values
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred {e}")


def create_predict(part_id, features_id, values, timestamps):
    try:
        now = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
        conn = get_connection()
        cur = conn.cursor()
        
        # SQL Query
        sql = """
        INSERT INTO dl_predict (id, part_id, features_id, date_time, pfi_value, status, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s,%s, %s)
        """
        
        # Iterasi dan eksekusi untuk setiap prediksi
        data_to_insert = []
        for value, timestamp in zip(values, timestamps):
            predict_id = str(uuid.uuid4())  # Generate a new UUID for each record
            value = float(value)
            data_to_insert.append((predict_id, part_id, features_id, timestamp, value, "normal", now, now))
        
        # Execute batch insert
        cur.executemany(sql, data_to_insert)
        # Commit perubahan
        conn.commit()
        print_log(f"Predictions successfully saved for part_id {part_id}, features_id {features_id}.")
    
    except Exception as e:
        print_log(f"An exception occurred: {e}")
        print(f"An exception occurred: {e}")
    
    finally:
        # Pastikan koneksi ditutup
        if cur:
            cur.close()
        if conn:
            conn.close()


def delete_predicts(part_id, features_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM dl_predict
            WHERE part_id = %s AND features_id = %s
            """,
            (part_id, features_id),
        )
        conn.commit()
        print_log(f"Predictions for part_id {part_id}, features_id {features_id} successfully deleted.")
    except Exception as e:
        print(f"An exception occurred while deleting predicts: {e}")
        print_log(f"An exception occurred: {e}")
      

