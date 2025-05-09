# sentiment_model_tf.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalAveragePooling1D, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class SentimentModel:
    def __init__(self, vocab_size=20000, max_length=200, embedding_dim=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.model = None

    def build_model(self):
        inputs = Input(shape=(self.max_length,), name='input')
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_length)(inputs)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = tf.keras.layers.Attention()([x, x])
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid', name='output')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

    def preprocess_data(self, texts, labels):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        return padded, np.array(labels)

    def train(self, x_train, y_train, x_val, y_val, epochs=5, batch_size=64):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val),
                       epochs=epochs, batch_size=batch_size, callbacks=[callback])

    def save_model(self, path='saved_models/sentiment_model.h5'):
        self.model.save(path)

    def load_model(self, path='saved_models/sentiment_model.h5'):
        self.model = tf.keras.models.load_model(path)

    def predict_sentiment(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        preds = self.model.predict(padded)
        return ['Positive' if p > 0.5 else 'Negative' for p in preds.flatten()]

if __name__ == "__main__":
    from tensorflow.keras.datasets import imdb
    from sklearn.model_selection import train_test_split

    print("Loading IMDB dataset...")
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = imdb.load_data(num_words=20000)
    word_index = imdb.get_word_index()
    index_word = {v+3: k for k, v in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"
    index_word[3] = "<UNUSED>"

    def decode_review(encoded):
        return ' '.join([index_word.get(i, '?') for i in encoded])

    texts_train = [decode_review(x) for x in x_train_raw]
    texts_test = [decode_review(x) for x in x_test_raw]

    print("Preprocessing data...")
    model = SentimentModel()
    x_train, y_train = model.preprocess_data(texts_train, y_train_raw)
    x_test, y_test = model.preprocess_data(texts_test, y_test_raw)

    print("Building model...")
    model.build_model()
    print("Training model...")
    model.train(x_train, y_train, x_test, y_test, epochs=5)

    print("Saving model...")
    model.save_model()
    print("Done.")
# predict.py

from sentiment_model_tf import SentimentModel
import os

# Load model and tokenizer
model = SentimentModel()
model.load_model("saved_models/sentiment_model.h5")

# 
sample_texts = [
    "I loved the movie, it was fantastic!",
    "The film was terrible and boring.",
    "An average experience, not bad but not good either."

]
predictions = model.predict_sentiment(sample_texts)

# 
for text, pred in zip(sample_texts, predictions):
    print(f"Text: {text}\nPredicted Sentiment: {pred}\n")
