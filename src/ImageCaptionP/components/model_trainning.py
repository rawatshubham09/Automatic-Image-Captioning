import os
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import dagshub
from tqdm import tqdm
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


from src.ImageCaptionP import logger
from src.ImageCaptionP.entity.config_entity import TrainningConfig
from src.ImageCaptionP.utils.common import load_bin,text_preprocessing, save_bin
from src.ImageCaptionP.components.datagenerator import CustomDataGenerator


class Training:
    def __init__(self, config: TrainningConfig):
        self.config = config

    def get_base_model(self):
        self.densenet_model = load_model(self.config.densnet_model_path)
        self.main_model = load_model(self.config.un_trained_main_model_path)
        logger.info("Base model and Densenet model loaded successfully.")

    def get_image_features(self):
        self.features = {}

        image_name_list = os.listdir(self.config.image_data_folder)

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

        logger.info("Image features generated successfully and saved.")
    
    def get_custom_generator(self):
        # Create a training data generator
        train = pd.read_csv(self.config.train_csv_file_path)
        valid = pd.read_csv(self.config.validation_csv_file_path)

        train = text_preprocessing(train)
        valid = text_preprocessing(valid)

        tokenizer = load_bin(self.config.tokenizer_path)

        self.train_generator = CustomDataGenerator(
            df=train,  # DataFrame containing training data
            X_col=self.config.x_col,  # Column name with image identifiers
            y_col=self.config.y_col,  # Column name with captions
            batch_size=self.config.params_batch_size,  # Number of samples per batch
            directory=self.config.image_data_folder,  # Path to the directory containing images
            tokenizer=tokenizer,  # Tokenizer to convert captions to sequences
            vocab_size=self.config.vocab_size,  # Total vocabulary size for one-hot encoding
            max_length=self.config.max_sent_length,  # Maximum length for input sequences
            features=self.features  # Pre-computed image features
        )

        logger.info("Train generators created successfully.")
        # Create a validation data generator
        self.validation_generator = CustomDataGenerator(
            df=valid,  # DataFrame containing validation data
            X_col=self.config.x_col,  # Column name with image identifiers
            y_col=self.config.y_col,  # Column name with captions
            batch_size=self.config.params_batch_size,  # Number of samples per batch
            directory=self.config.image_data_folder,  # Path to the directory containing images
            tokenizer=tokenizer,  # Tokenizer to convert captions to sequences
            vocab_size=self.config.vocab_size,  # Total vocabulary size for one-hot encoding
            max_length=self.config.max_sent_length,  # Maximum length for input sequences
            features=self.features  # Pre-computed image features
        )
        logger.info("Valid generators created successfully.")
    
    def get_callbacks(self):
        
        # Callback to save the model with the lowest validation loss
        model_name = "model.keras"  # Change the file extension to '.keras'
        self.checkpoint = ModelCheckpoint(
            model_name,  # Filepath where the model will be saved
            #filepath = self.config.model_checkpoint_file_path,
            monitor="val_loss",  # Monitor the validation loss during training
            mode="min",  # Save the model when the validation loss is minimized
            save_best_only=True,  # Save only the model with the best validation loss
            verbose=1  # Print a message when the model is saved
        )

        # Callback to stop training early if the validation loss does not improve
        self.earlystopping = EarlyStopping(
            monitor='val_loss',  # Monitor the validation loss
            min_delta=0,  # Minimum change in the monitored value to be considered an improvement
            patience=2,  # Number of epochs with no improvement after which training will be stopped
            verbose=1,  # Print a message when early stopping is triggered
            restore_best_weights=True  # Restore model weights from the epoch with the best validation loss
        )

        # Callback to reduce the learning rate if validation loss does not improve
        self.learning_rate_reduction = ReduceLROnPlateau(
            monitor='val_loss',  # Monitor the validation loss
            patience=2,  # Number of epochs with no improvement after which the learning rate will be reduced
            verbose=1,  # Print a message when the learning rate is reduced
            factor=0.2,  # Factor by which the learning rate will be reduced (new_lr = lr * factor)
            min_lr=1e-8  # Lower bound on the learning rate to avoid reducing it too much
        )
        logger.info("Callbacks created successfully.")
    
    def train_model(self):
        logger.info("Inside TrainModel function")
        self.main_model.compile(loss='categorical_crossentropy', optimizer='adam')

        mlflow.set_tracking_uri(self.config.mlflow_uri)

        with mlflow.start_run() as run:
            self.main_model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                validation_data=self.validation_generator,  # Validation data generator
                callbacks=[mlflow.keras.MlflowCallback(run, log_every_epoch=True),self.checkpoint,
                    self.earlystopping, self.learning_rate_reduction],
            )
        logger.info("Model trained successfully.")
        self.main_model.save(self.config.trained_main_model_path)
        #save_bin(self.main_model, self.config.trained_main_model_path)
        logger.info("Model saved successfully.")

