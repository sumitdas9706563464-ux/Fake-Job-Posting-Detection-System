# Fake Job Posting Detection System 🚫💼

An advanced NLP project using **TensorFlow/Keras** and **Streamlit** to identify fraudulent job postings.

## 📌 Project Overview
The goal of this project is to build a robust system that can automatically classify job postings as **Real** or **Fake**. This is a critical problem in the HR and recruitment industry, protecting job seekers from identity theft and financial scams.

### 💼 Business Impact
- **User Safety:** Protects users from potential scams.
- **Platform Trust:** Increases the credibility of job boards.
- **Efficiency:** Reduces the need for manual moderation by automating the screening process.

---

## 📊 Dataset
The project is designed to work with the **EMSCAD** (Employment Scam Aegean Dataset) which contains ~18,000 job descriptions.
- **Target:** `fraudulent` (0 for Real, 1 for Fake)
- **Features:** Job title, description, requirements, etc.
- **Class Imbalance:** Only ~5% of the data is fraudulent, requiring special handling.

---

## 🏗️ Project Structure
```text
├── data/               # Dataset (CSV)
├── models/             # Saved model and tokenizer
├── src/
│   ├── preprocess.py   # Text cleaning and tokenization
│   ├── train.py        # Model architecture and training
│   └── predict.py      # Inference pipeline
├── eda_results/        # EDA plots and charts
├── app.py              # Streamlit dashboard
├── Dockerfile          # Containerization
└── requirements.txt    # Dependencies
```

---

## 🧠 Model Architecture
We use a **Deep Learning** approach with:
1. **Embedding Layer:** Converts words into dense vectors of fixed size.
2. **SpatialDropout1D:** Helps prevent overfitting in sequences.
3. **Bidirectional LSTM (BiLSTM):** Captures context from both past and future words in the description.
4. **Dense Layers:** For classification with ReLU and Sigmoid activations.

### Handling Class Imbalance
Since fake jobs are rare, we use **Class Weights** during training to ensure the model pays more attention to the minority (fraudulent) class.

---

## 🚀 How to Run

### 1. Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (if needed)
python3 generate_data.py

# Train the model
export PYTHONPATH=$PYTHONPATH:.
python3 src/train.py

# Run the app
streamlit run app.py
```

### 2. Docker
```bash
docker build -t fake-job-detector .
docker run -p 8501:8501 fake-job-detector
```

---

## ☁️ Deployment Instructions

### Deploying to Render / Railway
1. **Connect GitHub:** Link your repository to the platform.
2. **Configuration:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py`
3. **Environment:** Ensure port `8501` is exposed or the platform's default port is used.

---

## 🔬 Advanced Topics

### 🔍 Model Explainability
To understand *why* a job is marked as fake, we can use:
- **LIME / SHAP:** To identify which specific words contributed most to the "Fake" prediction.
- **Keyword Analysis:** Highlighting words like "Urgent", "WhatsApp", "Bank Account", and "$$$" which often trigger the fraud detector.

### 🚀 Scaling for Production
- **FastAPI:** Wrap the model in a FastAPI backend for high-performance inference.
- **AWS SageMaker / Lambda:** Deploy the model as a scalable serverless endpoint.
- **Redis Cache:** Store predictions for common job descriptions to reduce latency.

### 📈 Future Improvements with BERT
Using a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model would significantly improve accuracy by leveraging deep linguistic context and understanding nuances that a standard LSTM might miss.

---

