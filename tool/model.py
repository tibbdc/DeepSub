'''
Author: DengRui
Date: 2024-01-20 13:15:08
LastEditors: DengRui
LastEditTime: 2024-01-20 13:23:06
FilePath: /DeepSub/trainer.ipynb
Description:  trainer
Copyright (c) 2024 by DengRui, All Rights Reserved. 
'''
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense,Dropout
from tool.att import Attention
from keras.layers import GRU, Bidirectional
import os

class GRUWithAttentionModel:
    def __init__(self, input_shape, num_classes, attention_size=32):
        """
        Initialization function for setting up model parameters.

        Args:
            input_shape (tuple): The shape of the input data, for example (batch_size, seq_len, feature_size).
            num_classes (int): The number of classes.
            attention_size (int, optional): The size of the attention mechanism. Defaults to 32.

        Returns:
            None
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.attention_size = attention_size
        self.model = self.build_model()

    def build_model(self):
        """
        Builds a model based on Bidirectional GRU and attention mechanism.

        Args:
            None

        Returns:
            model: The constructed model.
            
        """
        inputs = Input(shape=self.input_shape, name="input")
        gru = Bidirectional(GRU(128, dropout=0.5, return_sequences=True), name="bi-gru")(inputs)
        attention = Attention(self.attention_size, name='attention')(gru)
        attention_with_dropout = Dropout(0.1, name="attention_dropout")(attention)
        output = Dense(self.num_classes, activation='softmax', name="dense")(attention_with_dropout)
        model = Model(inputs, output)
        return model

    def compile_model(self):
        """
        Compile the model, configuring the optimizer, loss function, and evaluation metrics.

        Args:
            No parameters.

        Returns:
            No return value.
            
        """
        self.model.compile(optimizer=Adam(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, train_data, train_labels, validation_data, validation_labels, batch_size=1024, epochs=200):
        """
        Train the model using the provided data.

        Arguments:
            train_data (numpy.ndarray): Training data with shape (num_samples, num_features)
            train_labels (numpy.ndarray): Training labels with shape (num_samples, num_classes)
            validation_data (numpy.ndarray): Validation data with shape (num_samples, num_features)
            validation_labels (numpy.ndarray): Validation labels with shape (num_samples, num_classes)
            batch_size (int, optional): Batch size for training, defaults to 1024
            epochs (int, optional): Number of training epochs, defaults to 200

        Returns:
            history (History): Keras History object containing all information from the training process
            
        """
        history = self.model.fit(train_data, train_labels,
                                 validation_data=(validation_data, validation_labels),
                                 batch_size=batch_size, epochs=epochs)
        return history
    
    def save_model(self, path):
        """
        Save the model to a file.
        
        Args:
            path (str): The path of the saved model.
            
        Returns:
            None
            
        """
        model_dir = os.path.dirname(path)
    
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Directory {model_dir} was created.")
            
        self.model.save(path)
    


