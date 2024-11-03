from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import pickle
# Load the saved model
model = load_model("sentiment_model.h5")
print("Model loaded successfully!")
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
    else:
        text = ""  # Replace non-string or NaN with an empty string
    return text

# Load the tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)
print("Tokenizer loaded successfully!")
# Function to preprocess a single query
def preprocess_query(query):
    query = preprocess_text(query)  # Apply the same cleaning as training data
    sequence = tokenizer.texts_to_sequences([query])  # Convert to sequence
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    return padded_sequence

# Example queries to test
test_queries = ["I love the product, but the customer service is terrible.", "The service was exactly what I expected.", "The service was exactly what I expected."]
test_queries = input("Enter Query : ")
while test_queries != "End" :
    test_queries = input("Enter Query : ")
    if test_queries != "End":
        processed_query = preprocess_query(test_queries)
        prediction = model.predict(processed_query)
        sentiment = np.argmax(prediction)  # Get the sentiment label with the highest probability
        sentiment_label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        print(f"Query: {test_queries}")
        print(f"Predicted Sentiment: {sentiment_label[sentiment]}\n")
# Predict sentiment for each query
##for query in test_queries:
##    processed_query = preprocess_query(query)
##    prediction = model.predict(processed_query)
##    sentiment = np.argmax(prediction)  # Get the sentiment label with the highest probability
##    sentiment_label = {0: "Negative", 1: "Neutral", 2: "Positive"}
##    print(f"Query: {query}")
##    print(f"Predicted Sentiment: {sentiment_label[sentiment]}\n")
