# -*- coding: utf-8 -*-
"""
Refactored AlexNet Implementation
"""

import tensorflow as tf
from tensorflow.keras import Model, Input  # Changed: Imported Model and Input for Functional API usage
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Changed: Converted from a class inheriting 'Sequential' to a function that returns a 'Model'
def create_alexnet(input_shape, num_classes):
    
    # Changed: Explicit Input layer is now defined first (Functional API requirement), 
    # rather than passing 'input_shape' argument to the first Conv2D layer.
    inputs = Input(shape=input_shape)

    # Changed: Syntax switched from 'self.add(Layer)' to 'output = Layer(args)(input)' 
    # to explicitly chain layers together.
    
    # First Convolutional Layer
    x = Conv2D(96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    # Second Convolutional Layer
    # Changed: We continue passing the variable 'x' to chain the graph.
    x = Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    # Third, Fourth, and Fifth Convolutional Layers
    x = Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    # Flatten Layer
    x = Flatten()(x)

    # Fully Connected Layers
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Changed: The final layer output is captured in a specific variable 'outputs'
    outputs = Dense(num_classes, activation='softmax')(x)

    # Changed: We explicitly instantiate the Model object linking inputs to outputs,
    # instead of the model being implicit in the class structure.
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
input_shape = (224, 224, 3) 
num_classes = 1000 

# Changed: Calling the builder function instead of instantiating a class
model = create_alexnet(input_shape, num_classes)

model.summary()