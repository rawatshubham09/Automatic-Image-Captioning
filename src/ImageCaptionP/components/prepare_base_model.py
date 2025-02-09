import os
import json
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from box import ConfigBox
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Flatten, Layer
from tensorflow.keras.layers import Input, Dense, Reshape, Embedding, LSTM, Dropout, add, concatenate, Concatenate

from src.ImageCaptionP import logger
from src.ImageCaptionP.entity.config_entity import DataValidationConfig
from src.ImageCaptionP.utils.common import save_bin, save_json, text_preprocessing, save_yaml, read_yaml

class ModelBuilder:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.tokenizer = Tokenizer()
        self.vocab_size = 0
        self.max_length = 0
        
    #download densenet201 and generate image features
    def build_densenet_model_and_generate_image_feature(self):
        try:
            logger.info("Loading Densenet201 model...")
            # load densenet201 model
            model = DenseNet201(weights=self.config.params_weights)

            # This is done to extract image features instead of class predictions
            model = Model(inputs=model.input, outputs=model.layers[-2].output)

            # save densenet201 model
            save_bin(model, self.config.dense_model_path)
            logger.info("Densenet201 model loaded successfully.")

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e
    
    def train_tokenizer(self):
        try:
            logger.info("Training tokenizer...")

            df = pd.read_csv(self.config.captions_csv_file_path)
            # Preprocess the text data
            df = text_preprocessing(df)

            captions = df['caption'].tolist()
            # Fit the tokenizer on the preprocessed text data
            self.tokenizer.fit_on_texts(captions)

            self.vocab_size = len(self.tokenizer.word_index) + 1
            self.max_length = max(len(caption.split()) for caption in captions)

            # Load existing data
            confi_box = read_yaml(self.config.params_yaml_file_path)
            
            # Convert ConfigBox to dictionary
            existing_data = confi_box.to_dict()
            
            # New data to be added
            existing_data["VOCAB_SIZE"] = self.vocab_size
            existing_data["MAX_SENT_LENGTH"] = self.max_length
            
            # Save updated data back to the YAML file
            save_yaml(self.config.params_yaml_file_path, existing_data)
            # Save the tokenizer
            save_bin(self.tokenizer, self.config.tokerizer_path)

            logger.info("Tokenizer trained successfully and saved.")

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e
    
    def build_main_model(self):
        try:
            # Define two input layers
            logger.info("Building main model...")

            input1 = Input(shape=(1920,))  # Input for image features
            input2 = Input(shape=(self.max_length,))  # Input for the caption sequence

            # Image feature processing
            img_features = Dense(256, activation='relu')(input1)  # Fully connected layer to reduce dimensionality
            img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)  # Reshape to (1, 256) to concatenate with LSTM output

            # Caption (text) processing
            sentence_features = Embedding(self.vocab_size, 256, mask_zero=False)(input2)  # Embedding layer for input captions
            merged = concatenate([img_features_reshaped, sentence_features], axis=1)  # Concatenate image features with caption sequence
            sentence_features = LSTM(256)(merged)  # LSTM processes the combined features

            # Combine LSTM output with image features
            x = Dropout(0.5)(sentence_features)  # Dropout to prevent overfitting
            x = add([x, img_features])  # Skip connection to add image features to LSTM output
            x = Dense(128, activation='relu')(x)  # Fully connected layer
            x = Dropout(0.5)(x)  # Dropout for regularization
            output = Dense(self.vocab_size, activation='softmax')(x)  # Output layer with softmax activation to predict the next word

            # Create and compile the model
            caption_model = Model(inputs=[input1, input2], outputs=output)  # Define the model with two inputs and one output
            #caption_model.compile(loss='categorical_crossentropy', optimizer='adam')  # Compile the model with categorical crossentropy loss
            
            # save the model to disk
            save_bin(caption_model, self.config.main_model_path)
            
            print(caption_model.summary())
            logger.info("Main model created successfully.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e