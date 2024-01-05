import logging
import os

import coloredlogs
import urllib3
from dotenv import load_dotenv

from src.common.minio_fncts.minio_helpers import get_minio_client_secure_no_cert, upload_transitions_to_minio

coloredlogs.install(level='INFO')
load_dotenv()

MINIO_UPLOAD_BUCKET_NAME = os.getenv("MINIO_UPLOAD_BUCKET_NAME")
MINIO_UPLOAD_FROM = os.getenv("MINIO_UPLOAD_FROM")

if __name__ == "__main__":
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.warning("Disabled minio warnings")

    minio_client = get_minio_client_secure_no_cert()
    #minio_client.make_bucket("test-bucket2")
    local_filesystem_store_root_dir: str = MINIO_UPLOAD_FROM
    upload_transitions_to_minio(minio_client, MINIO_UPLOAD_BUCKET_NAME, local_filesystem_store_root_dir,
                                validate_paths=False)
