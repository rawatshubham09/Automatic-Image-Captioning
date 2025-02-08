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
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str