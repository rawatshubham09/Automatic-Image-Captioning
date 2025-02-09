# Ensure the ImageCaptionP module is in the Python path
from src.ImageCaptionP import logger
#from dotenv import load_dotenv
"""
from src.ImageCaptionP.pipeline.stage_01_data_ingestion import DataIngestionPipeline

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======================x")
except Exception as e:
    logger.exception(e)
    raise e
"""

"""
from src.ImageCaptionP.pipeline.stage_02_data_validation import DataValidationPipeline
STAGE_NAME = "Data Validation"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = DataValidationPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======================x")
except Exception as e:
    logger.exception(e)
    raise e
"""
STAGE_NAME = "Model Building"
from src.ImageCaptionP.pipeline.stage_03_model_builder import ModelBuilderPipeline
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = ModelBuilderPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======================x")
except Exception as e:
    logger.exception(e)
    raise e 