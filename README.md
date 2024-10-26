
<img width="1280" alt="readme-banner" src="https://github.com/user-attachments/assets/35332e92-44cb-425b-9dff-27bcf1023c6c">


# Sentence Emotion Analysis 🎯

## Basic Details
### Team Name: TechNoveL

### Team Members
- Team Lead: Helen Joji - [Your College]
- Member 2: Karthika Jayachandran - [Your College]
- Member 3: Jiya Reji - [Your College]

### Project Description
Our project performs sentiment analysis on user input sentences, categorizing emotions as Negative, Neutral, or Positive using a trained LSTM model.

### The Problem (that doesn't exist)
In a world where everyone can only communicate through emojis, the art of nuanced conversation is lost. We aim to revive emotional depth in text by analyzing sentiment!

### The Solution (that nobody asked for)
Introducing a chatbot that transforms bland text into insightful emotional analysis! Forget guessing your friends' moods—let our model do it for you!

## Technical Details
### Technologies/Components Used
For Software:
- Python
- TensorFlow/Keras
- Streamlit
- Pandas
- NLTK

### Implementation
For Software:
#### Installation
```bash
pip install pandas numpy tensorflow nltk streamlit
```

#### Run
```bash
streamlit run app.py
```

### Project Documentation
For Software:

#### Screenshots

![Screenshot 2024-10-26 113604](https://github.com/user-attachments/assets/3ffd690b-cf67-4009-b0c7-c2d1632dc708)

![Screenshot 2024-10-26 113636](https://github.com/user-attachments/assets/195dbbf2-44cd-4437-8ab4-8668a6351b25)


### Project Code
Here’s the code for training the sentiment analysis model:

```python
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, Input
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Read the data
train = pd.read_csv('train.csv', encoding='ISO-8859-1')
test = pd.read_csv('test.csv', encoding='ISO-8859-1')

# Data preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
train['text'] = train['text'].apply(preprocess_text)
test['text'] = test['text'].apply(preprocess_text)

# Tokenization
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(train['text'].values)

X = tokenizer.texts_to_sequences(train['text'])
X = pad_sequences(X, maxlen=100)
y = pd.get_dummies(train['sentiment']).values

# Build LSTM Model
text_input = Input(shape=(100,), name='text_input')
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(text_input)
text_lstm = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
text_lstm = Dropout(0.3)(text_lstm)
text_lstm = Bidirectional(LSTM(32))(text_lstm)
text_lstm = Dropout(0.5)(text_lstm)
output = Dense(y.shape[1], activation='softmax')(text_lstm)

# Define the model
model = Model(inputs=text_input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=0.00001)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), 
          callbacks=[early_stopping, lr_reduction])

# Function to predict sentiment from user input
def predict_sentiment(input_text):
    processed_text = preprocess_text(input_text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

# Main loop for user input
print("Starting sentiment analysis input loop...")
while True:
    user_input = input("Enter a sentence for sentiment analysis (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    sentiment = predict_sentiment(user_input)
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    print(f"Predicted Sentiment: {sentiment_labels[sentiment[0]]}")
```

And here’s the code for the Streamlit interface:

```python
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

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the pre-trained model
model_path = 'sentiment_model.h5'
model = load_model(model_path)

# Tokenization setup
tokenizer = Tokenizer(num_words=2000)

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
```

### Project Demo
#### Video
[Add your demo video link here]
*This video showcases how the sentiment analysis chatbot works in real-time.*

#### Additional Demos
[Add any extra demo materials/links]

## Team Contributions
- Helen Joji: Project management and frontend development.
- Karthika Jayachandran: Model training and backend integration.
- Jiya Reji: Data preprocessing and evaluation metrics.

---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProject--24-24?link=https%3A%2F%2Fwww.tinkerhub.org%2Fevents%2FQ2Q1TQKX6Q%2FUseless%2520Projects)
