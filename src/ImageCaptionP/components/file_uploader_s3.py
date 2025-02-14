import os
import boto3

from src.ImaceCaptionP import logger
from src.ImageCaptionP.components import S3Manager
from src.ImageCaptionP.entity.config_entity import S3DealerConfig

class s3Client:
    def __init__(self, config : S3DealerConfig):
        self.config = config
        self.s3 = S3Manager(self.config)

    def upload_best_models(self):

        try:
            #s3 = S3Manager(self.config)
            iteams_name_list = os.listdir(self.config.save_models_dir_path)

            bucket_name = self.config.s3_bucket_name
            if len(iteams_name_list) >= 3:
                self.s3.upload_file(self.config.save_model_path, bucket_name, "model.keras") #model
                self.s3.upload_file(self.config.save_tokenizer_path, bucket_name, "tokenizer.pkl") #token
                self.s3.upload_file(self.config.save_densenet_path, bucket_name, "densenet.keras") #dense
            else:
                logger.info("No best models found in the directory. retrain the model.")
                return
            
            logger.info("Models uploaded successfully to s3 bucket.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e
    
    def download_models(self):
        try:
            #s3 = S3Manager(self.config)
            bucket_name = self.config.s3_bucket_name
            self.s3.download_file(bucket_name, "model.keras", self.config.model_path)
            self.s3.download_file(bucket_name, "tokenizer.pkl", self.config.tokenizer_path)
            self.s3.download_file(bucket_name, "densenet.keras", self.config.densenet_path)

            logger.info("Models downloaded successfully from s3 bucket.")
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

