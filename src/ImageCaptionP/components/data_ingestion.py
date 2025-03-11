import os
import base64
import pandas as pd
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv
from src.ImageCaptionP.constants import DB_NAME, COLLECTION_NAME
from src.ImageCaptionP import logger
from src.ImageCaptionP.utils.common import create_directory
from src.ImageCaptionP.entity.config_entity import DataIngestionConfig
from src.ImageCaptionP.utils.common import save_yaml, read_yaml

#filePath = Path("src/ImageCaptionP/components/newFile.yaml")

class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config
        load_dotenv()

    def download_images_and_captions_from_mongodb(self):
        try:
            # Connect to MongoDB
            """#confi_box = read_yaml(filePath)
            
            # Convert ConfigBox to dictionary
            existing_data = confi_box.to_dict()
            
            ## if custome file is trainning:
            try:
                # New data to be added
            if existing_data["needTrained"] == "yes":
                client = MongoClient(existing_data["mongo_link"])
            else:"""
            client = MongoClient(self.config.mongo_URI)

            #existing_data["needTrained"] = "no"
            # Save updated data back to the YAML file
            """save_yaml(Path("params.yaml"), existing_data)
            except:
                client = MongoClient(self.config.mongo_URI)"""
            
            # Connecting to DB
            
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]
            logger.info('Connecting to MongoDB...')

            # Find all documents in the collection
            documents = collection.find()

            # Prepare a list to store the caption data for CSV
            csv_data = []

            create_directory([self.config.image_data_folder])
            # Loop through the documents and save the images
            for doc in documents:
                image_name = doc['image_name']
                encoded_image = doc['image_data']
                captions = doc['captions']
                image_path = os.path.join(self.config.image_data_folder, image_name)

                # Decode and save the image file
                with open(image_path, 'wb') as image_file:
                    image_file.write(base64.b64decode(encoded_image))
                
                # Add the captions to the CSV data list
                for caption in captions:
                    csv_data.append({'image': image_name, 'caption': caption})

            print("")
            # Save the CSV data to a file
            df = pd.DataFrame(csv_data)
            df.to_csv(self.config.csv_file_path, index=False)

            logger.info('mongo data Download complete!')
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e