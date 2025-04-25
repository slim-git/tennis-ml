from typing import Generator
from contextlib import contextmanager
import psycopg

from dotenv import load_dotenv
import os

load_dotenv()

@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in environment variables")
    
    conn = psycopg.connect(conninfo=DATABASE_URL)

    try:
        yield conn
    finally:
        conn.close()
