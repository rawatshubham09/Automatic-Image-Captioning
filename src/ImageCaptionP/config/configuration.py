import os
from dotenv import load_dotenv
from src.ImageCaptionP.constants import *
from pathlib import Path
from src.ImageCaptionP.utils.common import read_yaml, create_directory
from src.ImageCaptionP.entity.config_entity import (DataIngestionConfig,
                                                    DataValidationConfig,
                                                    PrepareBaseModelConfig)




class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directory([self.config.artifacts_root])
        load_dotenv()
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directory([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = Path(config.root_dir),
            mongo_URI = os.environ.get(config.mongo_uri),
            image_data_folder = Path(config.image_data_folder),
            csv_file_path = Path(config.csv_file_path)
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directory([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir = Path(config.root_dir),
            image_data_folder = Path(config.image_data_folder),
            csv_file_path = Path(config.csv_file_path),
            train_data_path = Path(config.train_data_path),
            validation_data_path = Path(config.validation_data_path),
            split_ratio = self.params.SPLIT_RATIO,
            bad_images_data_path = Path(config.bad_images_data_path),
            x_col = self.params.X_COL,
            y_col = self.params.Y_COL
        )

        return data_validation_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directory([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            params_image_size = self.param.IMAGE_SIZE,
            params_learning_rate = self.param.LEARNING_RATE,
            params_include_top = self.param.INCLUDE_TOP,
            params_weights = self.param.WEIGHTS,
        )

        return prepare_base_model_config