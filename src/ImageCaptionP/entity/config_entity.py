from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    mongo_URI: str
    image_data_folder: Path
    csv_file_path: Path