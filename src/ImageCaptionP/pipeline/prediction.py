import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.ImageCaptionP import logger
from src.ImageCaptionP.entity.config_entity import ImagePredictionsConfig
from src.ImageCaptionP.utils.common import idx_to_word, load_bin, save_bin, save_yaml

class ImageCaptionPredict:
    def __init__(self, config: ImagePredictionsConfig):
        self.config = config
    def get_models(self):
        try:
            self.densenet = load_model(self.config.densenet_path)
            self.model = load_model(self.config.model_path)
            self.tokenizer = load_bin(self.config.tokenizer_path)
            logger.info("Trained model, Densenet model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e
    
    def get_image_features(self, rawImage, image_name):
        try:
            self.features = {}

            #image shape
            image_shape = self.config.image_size

            # image preprocessing
            rawImage = rawImage.resize((image_shape, image_shape))  # Resize to 224*224
            rawImage = img_to_array(rawImage)
            rawImage = rawImage / 255.0  # Normalize pixel values to the range [0, 1]
            rawImage = np.expand_dims(rawImage, axis=0)  # Add an additional dimension to the image

            # feature prediction by densnet201
            feature = self.densenet.predict(rawImage, verbose=0)

            self.features[image_name] = feature

            logger.info("Image feature generated successfully and saved in prediction Class.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e
    def predict_caption(self, model, image_name, tokenizer, max_length):
        try:
            feature = self.features[image_name]  # Extract the feature for the given image
            in_text = "startseq"  # Start the caption generation with the starting token
            
            for i in range(max_length):  # Limit the length of the caption to max_length
                # Convert the input text so far into a sequence of integers
                sequence = tokenizer.texts_to_sequences([in_text])[0]  
                # Pad the sequence to ensure it has the required max_length
                sequence = pad_sequences([sequence], max_length)  
                
                # Predict the next word's index using the image feature and input sequence
                y_pred = model.predict([feature, sequence], verbose=0)  # Model predicts probabilities for each word in the vocabulary
                y_pred = np.argmax(y_pred)  # Choose the word with the highest probability
                
                word = idx_to_word(y_pred, tokenizer)  # Convert the predicted index back to a word
                
                if word is None:  # If no matching word is found, stop generation
                    break
                    
                in_text += " " + word  # Add the predicted word to the caption
                
                if word == 'endseq':  # If the 'endseq' token is predicted, end the generation
                    break
            #removing first and last word
            in_text = " ".join(in_text.split()[1:-1])
            
            logger.info(f"Text Predicted Successfully : {in_text}")

            return in_text  # Return the full caption as a string
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e
    
    def predict(self,rawImage,image_name):
        try:
            self.get_models()
            self.get_image_features(rawImage, image_name)
            out_put_text = self.predict_caption(self.model, image_name, self.tokenizer, self.config.max_sent_len)

            # Saving prediction to csv file
            if os.path.exists(self.config.predict_csv_path):
                """df = pd.read_csv(self.config.predict_csv_path)
                df = df.append({'image': image_name, 'caption': out_put_text}, ignore_index=True)
                df.to_csv(self.config.predict_csv_path, index=False)"""

                df = pd.read_csv(self.config.predict_csv_path)
                new_row = pd.DataFrame({'image': [image_name], 'caption': [out_put_text]})
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(self.config.predict_csv_path, index=False)
            else:
                df = pd.DataFrame({'image': [image_name], 'caption': [out_put_text]})
                df.to_csv(self.config.predict_csv_path, index=False)

            logger.info("Prediction saved successfully.")
            
            return out_put_text  # Return the full caption as a string
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e