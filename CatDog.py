# -*- coding: utf-8 -*-
"""
Refactored Cats vs Dogs Implementation (Functional API)
"""

import os
import zipfile
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# ==========================================
# 1. Data Preparation
# ==========================================

# NOTE: Update this path to where your actual zip file is located
local_zip = 'E:\\DL\\cats_and_dogs_filtered.zip' 
base_extract_dir = 'E:\\DL\\CatDog\\'

# Only try to unzip if the file actually exists to prevent immediate crashing
if os.path.exists(local_zip):
    print(f"Extracting {local_zip}...")
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall(base_extract_dir)
    zip_ref.close()
else:
    print(f"Warning: Zip file not found at {local_zip}. Please check path.")

# Define directory variables
base_dir = os.path.join(base_extract_dir, 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# ==========================================
# 2. Build Model (Functional API)
# ==========================================

def build_functional_model(input_shape):
    # Changed: Explicit Input layer defined first
    inputs = Input(shape=input_shape, name='img_input')

    # Changed: Chaining layers explicitly ( output = Layer(args)(input) )
    
    # Block 1
    x = Conv2D(16, (3,3), activation='relu', name='conv1')(inputs)
    x = MaxPooling2D(2,2, name='pool1')(x)

    # Block 2
    x = Conv2D(32, (3,3), activation='relu', name='conv2')(x)
    x = MaxPooling2D(2,2, name='pool2')(x)

    # Block 3
    x = Conv2D(64, (3,3), activation='relu', name='conv3')(x)
    x = MaxPooling2D(2,2, name='pool3')(x)

    # Flatten and Dense
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    
    # Output Layer
    # Changed: Output defined as a variable connected to the graph
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    # Changed: Model instantiation links inputs to outputs
    model = Model(inputs=inputs, outputs=outputs, name="CatDog_Functional")
    
    return model

# Instantiate model
input_shape = (150, 150, 3)
model = build_functional_model(input_shape)

# Changed: Updated 'lr' to 'learning_rate' for compatibility with newer TF versions
model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================
# 3. Data Generators
# ==========================================

train_datagen = ImageDataGenerator(rescale=1.0/255.)
test_datagen  = ImageDataGenerator(rescale=1.0/255.)

# Only flow if directories exist
if os.path.exists(train_dir):
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=20,
        class_mode='binary',
        target_size=(150, 150)
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        batch_size=20,
        class_mode='binary',
        target_size=(150, 150)
    )

    # ==========================================
    # 4. Training
    # ==========================================
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=100,
        epochs=15,
        validation_steps=50,
        verbose=2
    )

    # ==========================================
    # 5. Plotting Results
    # ==========================================
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

else:
    print("Training skipped because data directories were not found.")

# ==========================================
# 6. Feature Map Visualization
# ==========================================

# Only proceed if the model layers exist and we have data
if len(model.layers) > 0 and os.path.exists(train_cats_dir):
    
    # Changed: In Functional API, model.layers[0] is often the InputLayer.
    # We want outputs starting from the first Conv layer (index 1).
    successive_outputs = [layer.output for layer in model.layers[1:]]

    # Changed: Recreating a visualization model using Functional syntax
    # We use model.input (the input tensor of the original model)
    visualization_model = Model(inputs=model.input, outputs=successive_outputs)

    # Pick a random image
    cat_img_files = [os.path.join(train_cats_dir, f) for f in os.listdir(train_cats_dir)]
    dog_img_files = [os.path.join(train_dogs_dir, f) for f in os.listdir(train_dogs_dir)]
    
    if len(cat_img_files) > 0:
        img_path = random.choice(cat_img_files + dog_img_files)
        
        img = load_img(img_path, target_size=(150, 150))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        x /= 255.0

        successive_feature_maps = visualization_model.predict(x)
        layer_names = [layer.name for layer in model.layers[1:]]

        for layer_name, feature_map in zip(layer_names, successive_feature_maps):
            if len(feature_map.shape) == 4:
                # Standard visualization logic (kept from original)
                n_features = feature_map.shape[-1]
                size = feature_map.shape[1]
                display_grid = np.zeros((size, size * n_features))
                
                for i in range(n_features):
                    x_img = feature_map[0, :, :, i]
                    x_img -= x_img.mean()
                    if x_img.std() > 0: # Avoid division by zero
                        x_img /= x_img.std()
                    x_img *= 64
                    x_img += 128
                    x_img = np.clip(x_img, 0, 255).astype('uint8')
                    display_grid[:, i * size : (i + 1) * size] = x_img

                scale = 20. / n_features
                plt.figure(figsize=(scale * n_features, scale))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

# Removed the os.kill() command to prevent the python kernel from crashing unexpectedly.
print("Script finished.")