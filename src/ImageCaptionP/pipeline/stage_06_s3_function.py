import os
from src.ImageCaptionP.config.configuration import ConfigurationManager
from src.ImageCaptionP.components.file_uploader_s3 import s3Client
from src.ImageCaptionP import logger

STAGE_NAME = 'S3 Function stage'

class S3FunctionPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            config = ConfigurationManager()
            s3_config = config.get_s3_config()
            s3_model = s3Client(config=s3_config)

            s3_model.upload_best_models()
            logger.info("Best model get updated to S3 Bucket")

            folder_list = os.listdir(config.download_dir_path)
            if len(folder_list) != 3:
                s3_model.download_models()
            logger.info("Models from S3 bucket is downloaded") 

        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = S3FunctionPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======================x")
    except Exception as e:
        logger.exception(e)
        raise e