'''
Author: DengRui
Date: 2023-09-11 07:29:38
LastEditors: OBKoro1
LastEditTime: 2023-09-11 07:38:33
FilePath: /DeepSub/deepsub/att.py
Description: DengRui
Copyright (c) 2023 by DengRui, All Rights Reserved. 
'''

from keras import backend as K
from keras.layers import Layer, InputSpec
class Attention(Layer):
    def __init__(self, attention_size, **kwargs):
        """
            Initialization function for creating an instance of the Attention class.

        Args:
            attention_size (int): The size of the attention mechanism.
            **kwargs: Additional arguments that will be passed to the parent class initialization function.

        Returns:
            None
            
        """
        
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
            Builds the attention layer.
            
        Args:
            input_shape (tuple): The shape of the input tensor.
            **kwargs: Additional arguments that will be passed to the parent class initialization function.
        
        Returns:
            None
            
        """
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        """
            Call function for the attention layer.
        
        Args:
            x (tensor): The input tensor.
            mask (tensor): The mask tensor.
            **kwargs: Additional arguments that will be passed to the parent class initialization function.
            
        Returns:
            output (tensor): The output tensor.
            
        """
        et = K.tanh(K.dot(x, self.W) + self.b)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        """
            Compute mask function for the attention layer.
        
        Args:
            input (tensor): The input tensor.
            input_mask (tensor): The mask tensor.
            **kwargs: Additional arguments that will be passed to the parent class initialization function.
        
        Returns:
            None
            
        """
        return None

    def compute_output_shape(self, input_shape):
        """
            Compute output shape function for the attention layer.
        
        Args:
            input_shape (tuple): The shape of the input tensor.
            **kwargs: Additional arguments that will be passed to the parent class initialization function.
        
        Returns:
            output_shape (tuple): The shape of the output tensor.
            
        """
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        """
            Get configuration function for the attention layer.
            
        Args:
            **kwargs: Additional arguments that will be passed to the parent class initialization function.
        
        Returns:
            config (dict): The configuration dictionary.
            
        """
        config = {"attention_size": self.attention_size}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))