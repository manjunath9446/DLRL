# -*- coding: utf-8 -*-
"""
Refactored RNN Text Generator (Functional API)
Created on Tue Dec 16 2025

Description:
    Character-level text generation using SimpleRNN.
    Refactored to use the Keras Functional API with a Physics theme.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import SimpleRNN, Dense

# ==========================================
# 1. Data Preparation (Physics Theme)
# ==========================================

# Changed: New source text
text = "Energy cannot be created or destroyed only transformed from one form to another"

# Create mappings (Character <-> Index)
chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}
vocab_size = len(chars)

print(f"Total Characters: {len(text)}")
print(f"Unique Vocabulary: {vocab_size}")

# Configuration
seq_length = 5
hidden_units = 50 # Was 'text_len' in the original code

# Prepare Sequences
sequences = []
labels = []

for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[char] for char in seq])
    labels.append(char_to_index[label])

X = np.array(sequences)
y = np.array(labels)

# One-Hot Encoding
X_one_hot = tf.one_hot(X, vocab_size).numpy()
y_one_hot = tf.one_hot(y, vocab_size).numpy()

# ==========================================
# 2. Build Model (Functional API)
# ==========================================

def build_functional_rnn(input_shape, vocab_size, hidden_units):
    """
    Constructs the RNN using the Functional API.
    """
    # Changed: Explicit Input definition
    inputs = Input(shape=input_shape, name='input_char_sequence')

    # Changed: Layer chaining syntax -> output = Layer(args)(input)
    # Using 'relu' to match original code logic
    x = SimpleRNN(hidden_units, activation='relu', name='rnn_layer')(inputs)
    
    # Changed: Output layer
    outputs = Dense(vocab_size, activation='softmax', name='output_char_prob')(x)

    # Changed: Model instantiation
    model = Model(inputs=inputs, outputs=outputs, name="Physics_Text_Gen")
    return model

input_shape = (seq_length, vocab_size)
model = build_functional_rnn(input_shape, vocab_size, hidden_units)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================
# 3. Training
# ==========================================

print("\nStarting Training...")
# Using 150 epochs to ensure convergence on short text
model.fit(X_one_hot, y_one_hot, epochs=150, verbose=0) 
print("Training Complete.")

# ==========================================
# 4. Text Generation
# ==========================================

# Changed: Seed text relevant to the new theme
start_seq = "Energy cann" 
generated_text = start_seq
num_chars_to_generate = 50

print(f"\nSeed Sequence: '{start_seq}'")
print("-" * 30)

for i in range(num_chars_to_generate):
    # 1. Extract the last 'seq_length' characters
    current_slice = generated_text[-seq_length:]
    
    # 2. Convert characters to indices
    x_idx = [char_to_index[char] for char in current_slice]
    
    # 3. Reshape (1, seq_length) and One-Hot encode
    x_input = np.array([x_idx])
    x_one_hot = tf.one_hot(x_input, vocab_size)
    
    # 4. Predict
    prediction = model.predict(x_one_hot, verbose=0)
    
    # 5. Get the character with highest probability
    next_index = np.argmax(prediction)
    next_char = index_to_char[next_index]
    
    # 6. Append to result
    generated_text += next_char

print("Generated Text Result:")
print(generated_text)
print("-" * 30)