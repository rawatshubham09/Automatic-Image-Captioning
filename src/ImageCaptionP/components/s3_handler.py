import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

from src.ImageCaptionP import logger
from src.ImageCaptionP.entity.config_entity import S3DealerConfig


class S3Manager:
    def __init__(self, config: S3DealerConfig):
        self.config = config
        try:
            self.s3 = boto3.client('s3')  # Boto3 handles credentials
            logger.info("S3 client initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing S3 client: {e}")

    def upload_file(self, file_path, bucket_name, object_name=None):
        try:
            if object_name is None:
                object_name = os.path.basename(file_path) #if no object name, get the file name.
            self.s3.upload_file(file_path, bucket_name, object_name)
            logger.info(f"Successfully uploaded '{file_path}' to '{bucket_name}/{object_name}'.")
        except FileNotFoundError:
            logger.error(f"Error: File '{file_path}' not found.")
        except ClientError as e:
            logger.error(f"S3 error during upload: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}")

    def download_file(self, file_path, bucket_name, download_folder):
        try:
            filename = os.path.basename(file_path)
            local_path = os.path.join(download_folder, filename)
            os.makedirs(download_folder, exist_ok=True)
            self.s3.download_file(bucket_name, file_path, local_path)
            logger.info(f"Successfully downloaded '{file_path}' from '{bucket_name}' to '{local_path}'.")
        except FileNotFoundError:
            logger.error(f"Error: File '{file_path}' not found in bucket '{bucket_name}'.")
        except ClientError as e:
            logger.error(f"S3 error during download: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")