from src.ImageCaptionP.config.configuration import ConfigurationManager
from src.ImageCaptionP.components.data_validation import DataValidation
from src.ImageCaptionP import logger

STAGE_NAME = 'Data Validation stage'

class DataValidationPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.check_validation_of_data()
            data_validation.split_data()

        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = DataValidationPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======================x")
    except Exception as e:
        logger.exception(e)
        raise e