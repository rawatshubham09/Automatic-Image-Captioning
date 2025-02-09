import os
import re
import base64
import yaml
import json
import joblib
import pandas as pd
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError
from src.ImageCaptionP import logger

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
        return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def save_yaml(file_path: Path, data: dict):
    try:
        """Saves data to a YAML file."""
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file)
    except Exception as e:
        raise e

@ensure_annotations
def create_directory(path_to_directory: list, verbose=True):
    for path in path_to_directory:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"Directory: {path} created successfully")

@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path,"w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"JSON file: {path} saved successfully")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file: {path} loded successfully")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

def readImage(image_path:Path, img_size=224):
    img = load_img(image_path,color_mode='rgb',target_size=(img_size,img_size))
    img = img_to_array(img)
    img = img/255.
    return img

@ensure_annotations
def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: re.sub(r'[^A-Za-z\s]', '', x))
    data['caption'] = data['caption'].apply(lambda x: re.sub(r'\s+', ' ', x))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    data['caption'] = data['caption'].apply(lambda x: "startseq " + x + " endseq")
    return data