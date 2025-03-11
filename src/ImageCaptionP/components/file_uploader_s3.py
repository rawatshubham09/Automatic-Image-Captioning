import os
import boto3

from src.ImageCaptionP import logger
from src.ImageCaptionP.components.s3_handler import S3Manager
from src.ImageCaptionP.entity.config_entity import S3DealerConfig

class s3Client:
    def __init__(self, config: S3DealerConfig):
        self.config = config
        self.s3 = S3Manager(self.config)

    def upload_best_models(self):
        try:
            items_name_list = os.listdir(self.config.save_models_dir_path)
            bucket_name = self.config.s3_bucket_name

            if len(items_name_list) >= 3:
                # Ensure files exist before uploading, and use filename as object name.
                if os.path.exists(self.config.save_model_path):
                    self.s3.upload_file(self.config.save_model_path, bucket_name) #filename as object name.
                else:
                    logger.error(f"File not found: {self.config.save_model_path}")

                if os.path.exists(self.config.save_tokenizer_path):
                    self.s3.upload_file(self.config.save_tokenizer_path, bucket_name) #filename as object name
                else:
                    logger.error(f"File not found: {self.config.save_tokenizer_path}")

                if os.path.exists(self.config.save_densenet_path):
                    self.s3.upload_file(self.config.save_densenet_path, bucket_name) #filename as object name
                else:
                    logger.error(f"File not found: {self.config.save_densenet_path}")

                logger.info("Models uploaded successfully to s3 bucket.")
            else:
                logger.info("No best models found in the directory. Retrain the model.")
                return

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

    def download_models(self, download_folder="downloads"): #added download folder.
        try:
            bucket_name = self.config.s3_bucket_name

            # Ensure the download folder exists
            os.makedirs(download_folder, exist_ok=True)

            # Download files using the specified folder
            self.s3.download_file("model.keras", bucket_name, download_folder)
            self.s3.download_file("tokenizer.pkl", bucket_name, download_folder)
            self.s3.download_file("densenet.keras", bucket_name, download_folder)

            logger.info(f"Models downloaded successfully to '{download_folder}' from s3 bucket.")

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e