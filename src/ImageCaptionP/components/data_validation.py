import os
import pandas as pd
import numpy as np
from src.ImageCaptionP import logger
from src.ImageCaptionP.entity.config_entity import DataValidationConfig

# data validation
class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.bad_name = []
        self.good_name = []
        self.zero_size_name = []
    def check_validation_of_data(self):
        try:
            df = pd.read_csv(self.config.csv_file_path)

            # all image name in csv file
            csv_image_name = set(df[self.config.x_col])

            #name of all images in image folder
            image_name_list = os.listdir(self.config.image_data_folder)

            for img in image_name_list:
                name, ext = img.split('.')
                # get path of all images in image folder
                img_path = os.path.join(self.config.image_data_folder, img)
                if ext != "jpg" or img not in csv_image_name:
                    print(f"{img} has invalid extension. Only jpg files are allowed.")
                    self.bad_name.append(img)
                    #os.remove(img_path)
                elif os.path.getsize(img_path) == 0:
                    print(f"{img} has zero size.")
                    self.zero_size_name.append(img)
                    #os.remove(img_path)
                else:
                    self.good_name.append(img)
            
            logger.info("Bad file and image with zero size removed ")
            # Creating a text file and writing the lists to it
            with open(self.config.bad_images_data_path, "w") as file:
                file.write("Bad Image Names:\n")
                for image in self.bad_name:
                    file.write(f"{image}\n")
                
                file.write("\nImages with Zero Size:\n")
                for image in self.zero_size_name:
                    file.write(f"{image}\n")
                    
                file.write("\nGood Image Names:\n")
                for image in self.good_name:
                    file.write(f"{image}\n")

            logger.info("Text file created successfully.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e
    
    def split_data(self):
        try:
            # Splitting the data into train and validation sets
            df = pd.read_csv(self.config.csv_file_path)

            good_image_len = len(self.good_name)
        
            # Splitting the data into train and validation sets
            train_name = self.good_name[:int(good_image_len * self.config.split_ratio)]
            validation_name = self.good_name[int(good_image_len * self.config.split_ratio):]

            #df2 = df[df['image'].isin(list_images)]
            train_data = df[df[self.config.x_col].isin(train_name)]
            validation_data = df[df[self.config.x_col].isin(validation_name)]
            
            # Saving the train and validation data into separate csv files
            train_data.to_csv(self.config.train_data_path, index=False)
            validation_data.to_csv(self.config.validation_data_path, index=False)
            logger.info("Data Split Successfully.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

