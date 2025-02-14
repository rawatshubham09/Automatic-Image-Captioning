import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CustomDataGenerator(Sequence):
    def __init__(self, df, X_col, y_col, batch_size, directory, tokenizer, 
                 vocab_size, max_length, features, shuffle=True):
        """Initializes the generator with the given parameters."""
        self.df = df.copy()  # Make a copy of the DataFrame to avoid modifying the original one
        self.X_col = X_col  # Column name for image identifiers
        self.y_col = y_col  # Column name for captions
        self.directory = directory  # Directory where images are stored
        self.batch_size = batch_size  # Number of samples in each batch
        self.tokenizer = tokenizer  # Tokenizer to convert text to sequences
        self.vocab_size = vocab_size  # Size of the vocabulary for the captions
        self.max_length = max_length  # Maximum length of input sequences for padding
        self.features = features  # Pre-extracted features of the images
        self.shuffle = shuffle  # Whether to shuffle the data at the end of each epoch
        self.n = len(self.df)  # Total number of samples
    
    def on_epoch_end(self):
        """Shuffles the DataFrame at the end of each epoch if shuffle is True."""
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
    
    def __len__(self):
        """Returns the number of batches per epoch."""
        return self.n // self.batch_size  # Floor division to get the number of complete batches
    
    def __getitem__(self, index):
        """Generates one batch of data."""
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size, :]  # Get batch samples
        X1, X2, y = self.__get_data(batch)  # Get data for the batch
        return (X1, X2), y  # Return inputs and output as a tuple
    
    def __get_data(self, batch):
        """Generates data for a given batch of samples."""
        X1, X2, y = list(), list(), list()  # Initialize empty lists to store features, input sequences, and outputs
        
        images = batch[self.X_col].tolist()  # Get list of image identifiers in the batch
        
        for image in images:
            feature = self.features[image][0]  # Extract pre-computed image feature from the features dictionary
            captions = batch.loc[batch[self.X_col] == image, self.y_col].tolist()  # Get captions for the image
            for caption in captions:
                seq = self.tokenizer.texts_to_sequences([caption])[0]  # Convert caption to sequence of integers
                
                for i in range(1, len(seq)):
                    # Split the caption into input and output sequences
                    in_seq, out_seq = seq[:i], seq[i]  # Input sequence and the next word as the output
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]  # Pad input sequence to max_length
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]  # Convert output to one-hot encoding
                    
                    X1.append(feature)  # Append image feature as input
                    X2.append(in_seq)  # Append the input sequence
                    y.append(out_seq)  # Append the output (next word in sequence)
        
        # Convert lists to NumPy arrays for better performance
        X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                
        return X1, X2, y  # Return image features, input sequences, and output sequences
