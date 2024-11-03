import re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Clean and preprocess text
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
    else:
        text = ""  # Replace non-string or NaN with an empty string
    return text
data = pd.read_csv("Reddit_Data.csv")
data['clean_comment'] = data['clean_comment'].apply(preprocess_text)

# Tokenize and convert to sequences
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['clean_comment'])
sequences = tokenizer.texts_to_sequences(data['clean_comment'])

# Pad sequences to make them of equal length
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# Shift labels: -1 to 0, 0 to 1, and 1 to 2
data['category'] = data['category'].apply(lambda x: x + 1)

# Update labels after shifting
labels = data['category'].values

# Proceed with the train-test split as before
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)


print("Preprocessing complete and data is ready for model training.")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Model architecture
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=100),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')  # 3 units for the 3 classes (positive, neutral, negative)
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Save the model to a file
model.save("sentiment_model.h5")
print("Model saved as 'sentiment_model.h5'")

import pickle

# Save the tokenizer
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved as 'tokenizer.pickle'")
