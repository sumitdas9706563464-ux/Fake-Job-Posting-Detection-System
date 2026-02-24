import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.preprocess import TextPreprocessor

def build_model(vocab_size, embedding_dim=128, input_length=300):
    """
    Builds a BiLSTM model for binary classification.
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train():
    # 1. Load data
    df = pd.read_csv("data/fake_job_postings.csv")

    # 2. Preprocess
    preprocessor = TextPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    preprocessor.save_tokenizer()

    # 3. Handle Class Imbalance
    # Calculate class weights: total / (num_classes * count_of_class)
    counts = np.bincount(y_train)
    total = len(y_train)
    class_weight = {
        0: total / (2 * counts[0]),
        1: total / (2 * counts[1])
    }
    print(f"Class Weights: {class_weight}")

    # 4. Build Model
    vocab_size = preprocessor.max_words
    model = build_model(vocab_size)
    model.summary()

    # 5. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('models/job_classifier.keras', monitor='val_loss', save_best_only=True)
    ]

    # 6. Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=callbacks
    )

    # 7. Evaluate
    print("\nEvaluating model...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"AUC-ROC Score: {auc:.4f}")

    # 8. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('eda_results/confusion_matrix.png')
    print("Confusion matrix saved to eda_results/confusion_matrix.png")

    # 9. Save final model (redundant but good practice)
    # Note: .h5 is older, .keras is recommended now in Keras 3.
    # The requirement said .h5 or SavedModel. I'll use .h5 for compatibility as requested.
    model.save('models/job_classifier.h5')
    print("Model saved to models/job_classifier.h5")

if __name__ == "__main__":
    train()
