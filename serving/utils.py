import os

import structlog
from dotenv import load_dotenv
from google.cloud import storage

logger = structlog.get_logger()

load_dotenv(".env")


def load_model_from_gcs(
    bucket_name: str, source_blob_prefix: str, destination_dir: str
):
    """Downloads all files from a GCS prefix to a local directory."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(
        prefix=source_blob_prefix
    )  # List all files under the prefix

    os.makedirs(destination_dir, exist_ok=True)  # Create the destination directory

    for blob in blobs:
        # Construct the local file path
        local_file_path = os.path.join(destination_dir, os.path.basename(blob.name))
        blob.download_to_filename(local_file_path)
        logger.info(f"Downloaded {blob.name} to {local_file_path}.")
