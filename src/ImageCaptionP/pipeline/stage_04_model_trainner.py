from src.ImageCaptionP.config.configuration import ConfigurationManager
from src.ImageCaptionP.components.model_trainning import Training
from src.ImageCaptionP import logger

STAGE_NAME = 'Model Trainning stage'

class ModelTrainningPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            config = ConfigurationManager()
            model_trainning_config = config.get_training_config()
            model_trainer = Training(config=model_trainning_config)
            model_trainer.get_base_model()
            model_trainer.get_image_features()
            model_trainer.get_custom_generator()
            model_trainer.get_callbacks()
            model_trainer.train_model()

        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelTrainningPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======================x")
    except Exception as e:
        logger.exception(e)
        raise e