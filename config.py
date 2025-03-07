import os
import psycopg2  # type: ignore
from dotenv import load_dotenv  # type: ignore

load_dotenv()  # Load environment variables from .env file


class Config:
    PIWEB_API_URL = os.getenv("PIWEB_API_URL")
    PIWEB_API_USER = os.getenv("PIWEB_API_USER")
    PIWEB_API_PASS = os.getenv("PIWEB_API_PASS")
    PORT = os.getenv("PORT")

    DB_FETCH_HOST = os.getenv("DB_FETCH_HOST")
    DB_FETCH_USER = os.getenv("DB_FETCH_USER")
    DB_FETCH_PASS = os.getenv("DB_FETCH_PASS")
    DB_FETCH_NAME = os.getenv("DB_FETCH_NAME")
    DB_FETCH_PORT = os.getenv("DB_FETCH_PORT")

    DB_HOST = os.getenv("DB_HOST")
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_NAME = os.getenv("DB_NAME")
    DB_PORT = os.getenv("DB_PORT")
