import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

from src.ImageCaptionP import logger
from src.ImageCaptionP.entity.config_entity import S3DealerConfig


class S3Manager:
    def __init__(self, config : S3DealerConfig):
        self.config = config
        try:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.aws_region
            )
            logger.info("S3 client initialized successfully.")
        except NoCredentialsError:
            logger.error("Error: AWS credentials not available.")
        except PartialCredentialsError:
            logger.error("Error: Incomplete AWS credentials.")

    def upload_file(self, file_path, bucket_name, object_name):
        try:
            self.s3.upload_file(file_path, bucket_name, object_name)
            logger.info(f"Successfully uploaded {file_path} to {bucket_name}.")
        except FileNotFoundError:
            logger.error(f"Error: File {file_path} not found.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

    def download_file(self, file_path, bucket_name, download_folder):
        try:
            self.s3.download_file(bucket_name, file_path, f"{download_folder}/{file_path}")
            logger.info(f"Successfully downloaded {file_path} from {bucket_name}.")
        except FileNotFoundError:
            logger.error(f"Error: File {file_path} not found in bucket {bucket_name}.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        


"""
# Example usage:
class S3DealerConfig:
    def __init__(self, aws_access_key_id, aws_secret_access_key, aws_region):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region

config = S3DealerConfig('your-access-key-id', 'your-secret-access-key', 'your-region')
s3_manager = S3Manager(config)

# Example operations
s3_manager.upload_file('file1.txt', 'your-bucket-name')
s3_manager.download_file('file1.txt', 'your-bucket-name', 'download-directory')

"""
