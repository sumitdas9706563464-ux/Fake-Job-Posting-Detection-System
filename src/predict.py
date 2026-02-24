import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocess import TextPreprocessor
import os

class JobPredictor:
    def __init__(self, model_path='models/job_classifier.h5', tokenizer_path='models/tokenizer.pickle'):
        self.preprocessor = TextPreprocessor()
        self.preprocessor.load_tokenizer(tokenizer_path)
        # Load the model
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, text):
        """
        Predicts if a job description is fraudulent.
        Returns (is_fraudulent, probability)
        """
        # 1. Clean
        cleaned_text = self.preprocessor.clean_text(text)
        print(f"Cleaned Text: {cleaned_text}")

        # 2. Tokenize and Pad
        seq = self.preprocessor.tokenizer.texts_to_sequences([cleaned_text])
        print(f"Sequence: {seq}")
        padded = pad_sequences(seq, maxlen=self.preprocessor.max_len, padding='pre', truncating='pre')

        # 3. Predict
        prob = self.model.predict(padded)[0][0]
        is_fraud = bool(prob > 0.5)

        return is_fraud, float(prob)

if __name__ == "__main__":
    # Smoke test
    predictor = JobPredictor()

    test_texts = [
        "We are looking for a Software Engineer to join our team at TechNova Solutions. The ideal candidate will have strong communication skills and experience in the field. You will be responsible for managing projects and collaborating with cross-functional teams. We offer a competitive salary and great benefits.",
        "URGENT: Earn $5000 per week working from home. No experience needed. Apply now! Legit opportunity."
    ]

    for text in test_texts:
        is_fraud, prob = predictor.predict(text)
        print(f"\nText: {text[:50]}...")
        print(f"Fraudulent: {is_fraud} (Probability: {prob:.4f})")
