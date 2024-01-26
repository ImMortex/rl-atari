import io
import logging
import os
import tempfile
from typing import Text

import cv2
import numpy as np
import pandas as pd
from minio import Minio, S3Error
# from fio.ai.core.image import image_utils
from minio.commonconfig import CopySource
from numerize import numerize
from urllib3 import HTTPResponse


def object_exists(minio_client: Minio, bucket_name: Text, object_name: Text) -> bool:
    try:
        _ = minio_client.stat_object(bucket_name, object_name)
        return True
    except S3Error as error:
        if error.code == 'NoSuchKey':
            return False
        else:
            raise error


def object_prefix_exists(minio_client: Minio, bucket_name: Text, object_prefix: Text) -> bool:
    try:
        _ = next(minio_client.list_objects(bucket_name, prefix=object_prefix))
        return True
    except StopIteration:
        return False


def upload_object(minio_client: Minio, bucket_name: Text, object_name: Text, file_path: Text,

                  overwrite: bool = True, logs=False):
    already_exists = object_exists(minio_client, bucket_name, object_name)
    if overwrite or not already_exists:
        if logs:
            logging.info("Uploading File: " + file_path)
        minio_client.fput_object(bucket_name, object_name, file_path)

    if logs and already_exists:
        logging.info("Already on S3: " + object_name)

    return already_exists


def upload_image(minio_client: Minio, bucket_name: Text, object_name: Text, image: np.ndarray,

                 image_format: Text = 'jpg', overwrite: bool = False):
    with tempfile.TemporaryDirectory() as temp_path:
        file_path = os.path.join(temp_path, f'image.{image_format}')
        # image_utils.save_image(image, file_path)
        upload_object(minio_client, bucket_name, object_name, file_path, overwrite=overwrite)


def download_object(minio_client: Minio, bucket_name: Text, object_name: Text) -> HTTPResponse:
    return minio_client.get_object(bucket_name, object_name)


def download_data_frame(minio_client: Minio, bucket_name: Text, object_name: Text) -> pd.DataFrame:
    with download_object(minio_client, bucket_name, object_name) as o:
        with io.StringIO(o.read().decode('utf-8')) as string_io:
            return pd.read_csv(string_io)


def download_json(minio_client: Minio, bucket_name: Text, object_name: Text) -> Text:
    with download_object(minio_client, bucket_name, object_name) as o:
        return o.read().decode('utf-8')


def download_image(minio_client: Minio, bucket_name: Text, object_name: Text) -> np.ndarray:
    with download_object(minio_client, bucket_name, object_name) as o:
        buffer = np.frombuffer(o.read(), np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return image.astype(np.uint8)


def init_minio_client() -> Minio:
    """
    Returns a MinIOClient initialized from environment variables.
    Returns: The MinIOClient initialized from environment variables.
    """

    return Minio(os.environ['MINIO_URL'],
                 access_key=os.environ['MINIO_ACCESS_KEY'],
                 secret_key=os.environ['MINIO_SECRET_KEY'], secure=False)


def copy_path(minio_client: Minio, bucket_name: Text, source_path: Text, dest_path: Text):
    """
    Copies all objects from the given source location to the given destination location.

    Args:
        minio_client: The MinIO client that is ued to perform all MinIO requests.
        bucket_name: The name of the bucket.
        source_path: The prefix of source objects.
        dest_path: The destination prefix.
    """

    source_path = source_path.strip('/')
    dest_path = dest_path.strip('/')

    if object_prefix_exists(minio_client, bucket_name, dest_path):
        raise ValueError(f'destination path "{dest_path}" already exists')

    logging.info(f'prepare copy process from "{source_path}" to "{dest_path}"')
    object_count = 0
    object_size = 0
    dataset_objects = minio_client.list_objects(bucket_name, source_path, recursive=True)

    for dataset_object in dataset_objects:
        object_count += 1

        object_size += dataset_object.size

    logging.info(f'copy objects from "{source_path}" to "{dest_path}"')
    processed_size = 0
    dataset_objects = minio_client.list_objects(bucket_name, source_path, recursive=True)

    for dataset_object_id, dataset_object in enumerate(dataset_objects):
        source = CopySource(bucket_name, dataset_object.object_name)
        element_suffix = dataset_object.object_name[len(source_path):]
        target_name = dest_path + element_suffix
        minio_client.copy_object(bucket_name, target_name, source)
        processed_size += dataset_object.size
        logging.info(
            f'  copied {dataset_object_id + 1}/{object_count} objects: '
            f'{numerize.numerize(processed_size)}/{numerize.numerize(object_size)}')
