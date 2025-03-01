import os
from dotenv import load_dotenv
from src.ImageCaptionP.constants import *
from pathlib import Path
from src.ImageCaptionP.utils.common import read_yaml, create_directory
from src.ImageCaptionP.entity.config_entity import (DataIngestionConfig,
                                                    DataValidationConfig,
                                                    PrepareBaseModelConfig,
                                                    TrainningConfig,
                                                    BestModelConfig,
                                                    S3DealerConfig,
                                                    ImagePredictionsConfig,
                                                    FrountEndConfig)




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
            dense_model_path = Path(config.dense_model_path),                   #densenet201 h5 path
            main_model_path = Path(config.main_model_path),                     #custom model path
            image_feature_json_path = Path(config.image_feature_json_path),     #Image feature from densenet201
            image_data_folder = Path(config.image_data_folder),                 #image data folder
            captions_csv_file_path = Path(config.captions_csv_file_path),       #captions input csv file path
            tokerizer_path = Path(config.tokerizer_path),                       #tokenizer path
            params_image_size = self.params.IMAGE_SIZE,                          #[image_size]
            params_weights = self.params.WEIGHTS,
            params_yaml_file_path = Path(PARAMS_FILE_PATH),
            model_image_path = Path(config.model_image_path)

        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainningConfig:
        config = self.config.training

        create_directory([config.root_dir])

        training_config = TrainningConfig(
            root_dir = Path(config.root_dir),
            trained_main_model_path = Path(config.trained_main_model_path),
            un_trained_main_model_path = Path(config.un_trained_main_model_path),
            tokenizer_path = Path(config.tokenizer_path),
            densnet_model_path = Path(config.densnet_model_path),
            image_data_folder = Path(config.image_data_folder),
            train_csv_file_path = Path(config.train_csv_file_path),
            validation_csv_file_path = Path(config.validation_csv_file_path),
            model_checkpoint_file_path = Path(config.model_checkpoint_file_path),
            params_epochs = self.params.EPOCHS,
            params_batch_size = self.params.BATCH_SIZE,
            params_image_size = self.params.IMAGE_SIZE,
            vocab_size = self.params.VOCAB_SIZE,
            max_sent_length = self.params.MAX_SENT_LENGTH,
            mlflow_uri = os.environ.get(config.mlflow_uri),
            x_col = self.params.X_COL,
            y_col = self.params.Y_COL
        )

        return training_config
    
    def get_best_model_config(self) -> BestModelConfig:
        config = self.config.best_model_compare
        
        create_directory([config.root_dir])
        
        best_model_config = BestModelConfig(
            root_dir = Path(config.root_dir),
            best_model_path = Path(config.best_model_path),
            best_model_tokenizer_path = Path(config.best_model_tokenizer_path),
            old_model_path = Path(config.old_model_path),
            old_tokenizer_path = Path(config.old_tokenizer_path),
            image_data_folder = Path(config.image_data_folder),
            validation_csv_file_path = Path(config.validation_csv_file_path),
            bleu_score_yaml_file_path = Path(config.bleu_score_yaml_file_path),
            dense_model_path = Path(config.dense_model_path),
            winner_model_path =Path(config.winner_model_path),
            winner_tokenizer_path = Path(config.winner_tokenizer_path),
            winner_densenet_model_path = Path(config.winner_densenet_model_path),
            params_image_size = self.params.IMAGE_SIZE,
            max_sentence_length = self.params.MAX_SENT_LENGTH,
            best_max_sentence_length = self.params.BEST_MAX_SENT_LENGTH,
            x_col = self.params.X_COL,
            y_col = self.params.Y_COL
        )

        return best_model_config
    
    def get_s3_config(self):

        config = self.config.s3_pusher

        create_directory([config.root_dir])

        prepare_s3_config = S3DealerConfig(

            root_dir = Path(config.root_dir),
            s3_bucket_name = os.environ.get(config.s3_bucket_name),
            s3_region_name = os.environ.get(config.s3_region_name),
            aws_access_key_id = os.environ.get(config.aws_access_key_id),
            aws_secret_access_key = os.environ.get(config.aws_secret_access_key),
            save_models_dir_path = Path(config.save_models_dir),
            download_dir_path = Path(config.download_dir_path),
            model_path = Path(config.model_path),
            tokenizer_path = Path(config.tokenizer_path),
            densenet_path = Path(config.densenet_path),
            save_model_path = Path(config.save_model_path),
            save_tokenizer_path = Path(congig.save_tokenizer_path),
            save_densenet_path = Path(congig.save_densenet_path)
        )
    
        return prepare_s3_config
    
    def get_prediction_config(self):
        config = self.config.predictions
        
        create_directory([config.root_dir])
        
        prediction_config = ImagePredictionsConfig(
            root_dir = Path(config.root_dir),
            image_folder_path = Path(config.root_dir),
            predict_csv_path = Path(config.predict_csv_path),
            tokenizer_path = Path(config.tokenizer_path),
            densenet_path = Path(config.densenet_path),
            model_path = Path(config.model_path),
            max_sent_len = self.params.BEST_MAX_SENT_LENGTH,
            image_size = self.params.IMAGE_SIZE,
            x_col = self.params.X_COL,
            y_col = self.params.Y_COL,
        )
        
        return prediction_config
    
    def get_frountend_config(self):
        config = self.config.frountend

        create_directory([config.artifact_dir])

        frontend_config = FrountEndConfig(
            artifact_dir = Path(config.artifact_dir),
            image_folder = Path(config.image_folder),
            log_file_path = Path(config.log_file_path)
        )

        return frontend_config
        