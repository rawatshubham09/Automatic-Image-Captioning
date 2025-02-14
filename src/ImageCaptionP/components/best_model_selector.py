import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence, to_categorical
from nltk.translate.bleu_score import sentence_bleu
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.ImageCaptionP import logger
from src.ImageCaptionP.entity.config_entity import BestModelConfig
from src.ImageCaptionP.utils.common import idx_to_word, load_bin, save_bin, save_yaml



class BestModel:
    def __init__(self,config : BestModelConfig):
        self.config = config
    
    def get_base_model(self):
        try:
            self.densenet_model = load_model(self.config.dense_model_path)
            self.old_model = load_model(self.config.old_model_path)
            self.best_model = load_model(self.config.best_model_path)
            self.old_tokenizer = load_bin(self.config.old_tokenizer_path)
            self.best_tokenizer = load_bin(self.config.best_model_tokenizer_path)
            logger.info("Trained model and Densenet model loaded successfully.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

    def get_image_features(self):
        try:
            self.features = {}

            self.df = pd.read_csv(self.config.validation_csv_file_path)
            image_name_list = self.df[self.config.x_col].unique().tolist()

            image_shape = self.config.params_image_size

            for image in tqdm(image_name_list):  
                # Load the image from the specified path and resize it to the required size (224x224)
                img = load_img(os.path.join(self.config.image_data_folder, image), target_size=(image_shape,image_shape))  
                # Convert the image into a NumPy array (from PIL image to array)
                img = img_to_array(img)  
                # Normalize pixel values to the range [0, 1]
                img = img / 255.  
                # Add an additional dimension to the image to make it compatible with the model's expected input shape
                # The model expects input in the form (batch_size, height, width, channels), so we add an extra batch dimension
                img = np.expand_dims(img, axis=0)  
                # Use the feature extraction model to generate a feature vector for the image
                feature = self.densenet_model.predict(img, verbose=0)  
                # Store the feature vector in the 'features' dictionary, using the image name as the key
                self.features[image] = feature 

            logger.info("Image features generated successfully and saved in BestModel Class.")
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

            return in_text  # Return the full caption as a string
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e
    

    def evaluate_model(self, model, tokenizer, max_length):
        try:
            image_name_list = self.df[self.config.x_col].unique().tolist()
            total_image_length = len(image_name_list)

            bleu_score = 0

            # loop each image to fing the blue score.
            for img_name in tqdm(image_name_list):
                predicted_caption = self.predict_caption(model, img_name, tokenizer, max_length)
                actual_caption = self.df[self.df[self.config.x_col] == img_name][self.config.y_col].values[0]
                actual_caption = actual_caption.split()
                predicted_caption = predicted_caption.split()
                bleu_score += sentence_bleu([actual_caption], predicted_caption)

            logger.info(f"blue_score for model with max len : {max_length} is: {bleu_score}")
            return bleu_score

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

    def get_best_model(self):
        try:
            # calculate the bleu score for both models
            logger.info("Calculating Blue Score for both old and cloud model...")
            # old model
            old_bleu_score = self.evaluate_model(self.old_model, self.old_tokenizer, self.config.max_sentence_length)
            # best model
            best_bleu_score = self.evaluate_model(self.best_model, self.best_tokenizer, self.config.best_max_sentence_length)

            data = {"OLD_MODEL_BLEU_SCORE": old_bleu_score,
                    "CLOUD_MODEL_BLEU_SCORE": best_bleu_score,
                    "BEST_MAX_SENT_LENGTH": self.config.max_sentence_length}
            # save best mode and tokenizer model to location
            if old_bleu_score > best_bleu_score + 0.25:
                logger.info(f">>>>>>>>>>>>>>>>> winner model updated with old model")
                self.old_model.save(self.config.winner_model_path)
                shutil.copy(self.config.old_tokenizer_path,self.config.winner_tokenizer_path)
                shutil.copy(self.config.dense_model_path, self.config.winner_densenet_model_path)
                save_yaml(self.config.bleu_score_yaml_file_path,data)
            else:
                logger.info(f">>>>>>>>>>>>>> winner model is cloud model")
                self.best_model.save(self.config.winner_model_path)
                shutil.copy(self.config.best_model_tokenizer_path,self.config.winner_tokenizer_path)
                shutil.copy(self.config.dense_model_path, self.config.winner_densenet_model_path)
            logger.info(f"Best model is saved to folder : artifacts/best_model")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

    