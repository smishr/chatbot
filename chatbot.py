# pip3 install tensof
import tensorflow as tf
import numpy as np
import random
import json

# Load the intents file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Create the training data
training_data = []
output_data = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = pattern.split()
        training_data.append(words)
        output_data.append(intent['tag'])

# Create the vocabulary
vocabulary = sorted(set([word for words in training_data for word in words]))

# Create the word index dictionary
word_index = {word: index for index, word in enumerate(vocabulary)}

# Create the output index dictionary
output_index = {output: index for index, output in enumerate(sorted(set(output_data)))}

# Create the training data matrix
training_data_matrix = np.zeros((len(training_data), len(vocabulary)), dtype=np.int32)

for row, words in enumerate(training_data):
    for word in words:
        training_data_matrix[row, word_index[word]] = 1

# Create the output data matrix
output_data_matrix = np.zeros((len(output_data), len(output_index)), dtype=np.int32)

for row, output in enumerate(output_data):
    output_data_matrix[row, output_index[output]] = 1

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(vocabulary),), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(output_index), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(training_data_matrix, output_data_matrix, epochs=1000, batch_size=8)

# Save the model
model.save('sales_chatbot_model.h5')
