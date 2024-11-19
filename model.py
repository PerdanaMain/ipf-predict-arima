from database import get_connection
from datetime import datetime
from log import print_log
import pytz


def get_all_tags():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT id, web_id FROM dl_ms_tag")
        tags = cur.fetchall()
        cur.close()
        conn.close()
        return tags
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred: {e}")


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


def get_values(tag_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT tag_id, time_stamp, value 
            FROM dl_value_tag 
            WHERE tag_id = %s
            """,
            (tag_id,),
        )
        values = cur.fetchall()
        cur.close()
        conn.close()
        print("Data fetched successfully, count: ", len(values))
        return values
    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred {e}")


def create_predict(tag_id, values, timestamps):
    try:
        now = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
        conn = get_connection()
        cur = conn.cursor()
        sql = "INSERT INTO dl_predict_tag (tag_id, time_stamp, value, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)"
        cur.executemany(
            sql,
            [
                (tag_id, timestamps[i], float(values[i]), now, now)
                for i in range(len(values))
            ],
        )
        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred {e}")

def delete_predicts():
    try:
        conn = get_connection()
        cur = conn.cursor()
        sql = "DELETE FROM dl_predict_tag"
        cur.execute(sql)
        conn.commit()
        cur.close()
        conn.close()

        print_log("All prediction data has been deleted.")

    except Exception as e:
        print_log(f"An exception occurred {e}")
        print(f"An exception occurred {e}")
