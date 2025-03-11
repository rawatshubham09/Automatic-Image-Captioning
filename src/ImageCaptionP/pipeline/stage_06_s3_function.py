import os
from src.ImageCaptionP.config.configuration import ConfigurationManager
from src.ImageCaptionP.components.file_uploader_s3 import s3Client
from src.ImageCaptionP import logger

STAGE_NAME = 'S3 Download Stage'
EXPECTED_DOWNLOAD_FILES = 3

class S3FunctionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            s3_config = config.get_s3_config()
            s3_model = s3Client(config=s3_config)
            download_folder = "downloads" #set download folder.

            if os.path.exists(download_folder):
                folder_list = os.listdir(download_folder)
                if len(folder_list) != EXPECTED_DOWNLOAD_FILES:
                    logger.info("Starting model download from S3...")
                    s3_model.download_models(download_folder) #pass the folder
                    logger.info(f"Models from S3 bucket downloaded to '{download_folder}'")
                else:
                    logger.info(f"Download directory '{download_folder}' already contains the expected files.")
            else:
                logger.warning(f"Download directory '{download_folder}' does not exist.")
                logger.info("Starting model download from S3...")
                s3_model.download_models(download_folder) #pass the folder.
                logger.info(f"Models from S3 bucket downloaded to '{download_folder}'")

        except Exception as e:
            logger.exception(e)
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