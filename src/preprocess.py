import pandas as pd
import numpy as np
import re
import pickle
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class TextPreprocessor:
    def __init__(self, max_words=10000, max_len=300):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """
        Cleans the input text by:
        - Lowercasing
        - Removing special characters and numbers
        - Removing stopwords
        - Stripping extra whitespaces
        """
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize and remove stopwords
        words = word_tokenize(text)
        cleaned_words = [w for w in words if w not in self.stop_words]

        return " ".join(cleaned_words)

    def prepare_data(self, df, text_column='description', target_column='fraudulent', test_size=0.2):
        """
        Full pipeline: cleaning, tokenization, padding, and split.
        """
        print("Cleaning text...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)

        X = df['cleaned_text']
        y = df[target_column]

        # Split first to avoid data leakage (fit tokenizer only on train)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        print("Tokenizing...")
        self.tokenizer.fit_on_texts(X_train_raw)

        X_train_seq = self.tokenizer.texts_to_sequences(X_train_raw)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test_raw)

        print("Padding...")
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len, padding='pre', truncating='pre')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len, padding='pre', truncating='pre')

        return X_train_pad, X_test_pad, y_train, y_test

    def save_tokenizer(self, path='models/tokenizer.pickle'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer saved to {path}")

    def load_tokenizer(self, path='models/tokenizer.pickle'):
        with open(path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        print(f"Tokenizer loaded from {path}")

if __name__ == "__main__":
    # Test the preprocessor
    df = pd.read_csv("data/fake_job_postings.csv")
    preprocessor = TextPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    preprocessor.save_tokenizer()
