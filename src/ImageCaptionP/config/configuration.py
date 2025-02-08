from ImageCaptionP.constants import *
from ImageCaptionP.utils.common import read_yaml, create_directory
from ImageCaptionP.entity.config_entity import DataIngestionConfig



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directory([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directory([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            mongo_URI = os.environ.get(config.mongo_uri),
            image_data_folder = config.image_data_folder,
            csv_file_path = config.csv_file_path
        )

        return data_ingestion_config