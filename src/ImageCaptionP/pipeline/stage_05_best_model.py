from src.ImageCaptionP.config.configuration import ConfigurationManager
from src.ImageCaptionP.components.best_model_selector import BestModel
from src.ImageCaptionP import logger

STAGE_NAME = 'Best Model Selecting stage'

class BestModelSelectingPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            config = ConfigurationManager()
            best_model_config = config.get_best_model_config()
            best_model = BestModel(config=best_model_config)
            best_model.get_base_model()
            best_model.get_image_features()
            best_model.get_best_model()

        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = BestModelSelectingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======================x")
    except Exception as e:
        logger.exception(e)
        raise e