from database import get_connection


def get_all_tags(limit=10):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT id, web_id FROM dl_ms_tag limit {limit}")
        tags = cur.fetchall()
        cur.close()
        conn.close()
        return tags
    except Exception as e:
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
        print(f"An exception occurred {e}")
