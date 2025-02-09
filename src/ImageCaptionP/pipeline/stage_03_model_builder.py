from src.ImageCaptionP.config.configuration import ConfigurationManager
from src.ImageCaptionP.components.prepare_base_model import ModelBuilder
from src.ImageCaptionP import logger

STAGE_NAME = 'Model Building stage'

class ModelBuilderPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            config = ConfigurationManager()
            model_builder_config = config.get_prepare_base_model_config()
            model_builder = ModelBuilder(config=model_builder_config)
            model_builder.build_densenet_model_and_generate_image_feature()
            model_builder.train_tokenizer()
            model_builder.build_main_model()

        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelBuilderPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======================x")
    except Exception as e:
        logger.exception(e)
        raise e