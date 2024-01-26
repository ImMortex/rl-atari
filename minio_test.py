import logging
import os

import urllib3
from minio import Minio
from dotenv import load_dotenv

def test_minio_connection():
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.warning("Disabled minio warnings")
    load_dotenv()

    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
    MINIO_ADDRESS = os.getenv("MINIO_ADDRESS")
    MINIO_PORT = os.getenv("MINIO_PORT")

    client = Minio(
        MINIO_ADDRESS + ":" + MINIO_PORT,
        secure=True,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        cert_check=False
    )
    print("available minio buckets")
    for bucket in client.list_buckets():
        print(bucket.name)

if __name__ == "__main__":
    test_minio_connection()
