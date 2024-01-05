import logging
import traceback

import coloredlogs
import urllib3

from config.train_config import get_train_config
from src.common.minio_fncts.minio_helpers import get_minio_client_secure_no_cert, upload_transitions_to_minio

coloredlogs.install(level='INFO')

def upload_transitions_to_minio_bucket():
    finished = False
    while not finished:
        try:
            train_config = get_train_config()
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logging.warning("Disabled minio warnings")

            minio_client = get_minio_client_secure_no_cert()
            #minio_client.make_bucket("test-bucket2")
            upload_transitions_to_minio(minio_client, train_config["bucket_name"], train_config["local_filesystem_store_root_dir"],
                                               validate_paths=False)
            finished = True
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
