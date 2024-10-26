import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model_path = 'sentiment_model.h5'
model = load_model(model_path)

# Tokenization setup (ensure this matches the one used during training)
tokenizer = Tokenizer(num_words=2000)
# Here you might want to load the tokenizer if saved, or redefine it based on your training data.

# Data preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Function to predict sentiment from user input
def predict_sentiment(input_text):
    processed_text = preprocess_text(input_text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

# Streamlit interface
st.title("Sentiment Analysis Chatbot")
user_input = st.text_area("Enter a sentence for sentiment analysis:")

if st.button("Analyze"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        st.success(f"Predicted Sentiment: {sentiment_labels[sentiment[0]]}")
    else:
        st.warning("Please enter a sentence.")










