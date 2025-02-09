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

