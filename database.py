import psycopg2 # type: ignore
import os
from dotenv import load_dotenv # type: ignore

load_dotenv()


def get_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_MAIN_HOST"),
            database=os.getenv("DB_MAIN_NAME"),
            user=os.getenv("DB_MAIN_USER"),
            password=os.getenv("DB_MAIN_PASS"),
        )
        return conn
    except Exception as e:
        print(f"An exception occurred: {e}")
