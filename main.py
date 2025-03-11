# Ensure the ImageCaptionP module is in the Python path
from src.ImageCaptionP import logger
import warnings
#from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# run
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

STAGE_NAME = "Model Trainning"
from src.ImageCaptionP.pipeline.stage_04_model_trainner import ModelTrainningPipeline
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = ModelTrainningPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======================x")
except Exception as e:
    logger.exception(e)
    raise e 



STAGE_NAME = "Select Best Model"
from src.ImageCaptionP.pipeline.stage_05_best_model import BestModelSelectingPipeline
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
    obj = BestModelSelectingPipeline()
    obj.main()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======================x")
except Exception as e:
    logger.exception(e)
    raise e 


STAGE_NAME = "Select Best Model"
from src.ImageCaptionP.pipeline.stage_06_s3_function import S3FunctionPipeline
if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = S3FunctionPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx======================x")
    except Exception as e:
        logger.exception(e)
        raise e

logger.info(">>>>>>>>>>>>>>>>>>>> Training Completed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
