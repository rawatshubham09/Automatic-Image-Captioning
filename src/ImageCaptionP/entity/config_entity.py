from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    mongo_URI: str
    image_data_folder: Path
    csv_file_path: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    image_data_folder: Path
    csv_file_path: Path
    train_data_path: Path
    validation_data_path: Path
    bad_images_data_path: Path
    split_ratio: float
    x_col: str
    y_col: str

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    dense_model_path: Path
    main_model_path: Path
    image_feature_json_path: Path
    image_data_folder: Path
    tokerizer_path: Path
    captions_csv_file_path: Path
    model_image_path: Path
    params_yaml_file_path: Path
    params_image_size: list
    params_weights: str

@dataclass(frozen=True)
class TrainningConfig:
    root_dir: Path
    trained_main_model_path: Path
    tokenizer_path: Path
    densnet_model_path: Path
    image_data_folder: Path
    train_csv_file_path: Path
    validation_csv_file_path: Path
    model_checkpoint_file_path: Path
    un_trained_main_model_path: Path
    mlflow_uri: str
    params_epochs: int
    params_batch_size: int
    params_image_size: int
    vocab_size: int
    max_sent_length: int
    x_col: str
    y_col: str

@dataclass(frozen=True)
class BestModelConfig:
    root_dir: Path
    best_model_path: Path       # cloud downloaded model path
    best_model_tokenizer_path: Path
    old_model_path: Path        # model trained on local data
    old_tokenizer_path: Path
    image_data_folder: Path
    validation_csv_file_path: Path
    bleu_score_yaml_file_path: Path
    dense_model_path: Path
    winner_model_path: Path        # this contain best model after conparision
    winner_tokenizer_path: Path
    winner_densenet_model_path: Path
    params_image_size: int 
    max_sentence_length: int
    best_max_sentence_length: int
    x_col: str
    y_col: str

@dataclass(frozen=True)
class S3DealerConfig:
    root_dir: Path
    s3_bucket_name: str
    s3_region_name: str
    aws_access_key_id: str
    aws_secret_access_key: str
    save_models_dir_path: Path
    download_dir_path: Path
    model_path: Path
    tokenizer_path: Path
    densenet_path: Path
    save_model_path: Path
    save_tokenizer_path: Path
    save_densenet_path: Path

@dataclass(frozen=True)
class ImagePredictionsConfig:
    root_dir: Path
    image_folder_path: Path
    predict_csv_path: Path
    tokenizer_path: Path
    densenet_path: Path
    model_path: Path
    max_sent_len: int
    image_size: int
    x_col: str
    y_col: str