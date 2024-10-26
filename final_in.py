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

# Read the data (update the path to your CSV files)
train = pd.read_csv(r'C:\Users\Karthika\Documents\train.csv', encoding='ISO-8859-1')
test = pd.read_csv(r'C:\Users\Karthika\Documents\test.csv', encoding='ISO-8859-1')

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
model.fit(X_train, y_train, epochs=5,batch_size=32, validation_data=(X_val, y_val), 
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
    
    # Map predicted class to sentiment labels
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    print(f"Predicted Sentiment: {sentiment_labels[sentiment[0]]}")
