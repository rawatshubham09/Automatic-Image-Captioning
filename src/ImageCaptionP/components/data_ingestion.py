import os
from ImageCaptionP.constants import *
from ImageCaptionP import logger
from ImageCaptionP.utils.common import create_directory
from ImageCaptionP.entity.config_entity import (DataIngestionConfig)

class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.config = config

    def download_images_and_captions_from_mongodb(self):
        try:
            # Connect to MongoDB
            client = MongoClient(self.config.mongo_URI)
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

            # Save the CSV data to a file
            df = pd.DataFrame(csv_data)
            df.to_csv(self.config.csv_file_path, index=False)

            logger.info('mongo data Download complete!')
        except Exception as e:
            raise e