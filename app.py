import streamlit as st
import os
from src.predict import JobPredictor

# Set page config
st.set_page_config(page_title="Fake Job Detection", page_icon="🚫", layout="wide")

# Initialize Predictor
@st.cache_resource
def get_predictor():
    return JobPredictor()

predictor = get_predictor()

# App Title
st.title("🚫 Fake Job Posting Detection System")
st.markdown("""
This system uses a **Bidirectional LSTM (Deep Learning)** model to classify job postings as **Real** or **Fake**.
It analyzes the textual content of the job description to identify suspicious patterns common in fraudulent postings.
""")

# Sidebar info
st.sidebar.title("About the Project")
st.sidebar.info("""
- **Model:** BiLSTM (TensorFlow)
- **NLP:** Keras Tokenizer, Stopword removal
- **Goal:** Protect job seekers from scams.
""")

# Input Area
st.subheader("Enter Job Description")
job_text = st.text_area("Paste the job description here:", height=300,
                        placeholder="e.g., We are looking for a Software Engineer to join our team...")

if st.button("Analyze Job Posting"):
    if job_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            is_fraud, prob = predictor.predict(job_text)

            # Display results
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Prediction", "FRAUDULENT" if is_fraud else "REAL")

            with col2:
                st.metric("Confidence Score", f"{prob if is_fraud else 1-prob:.2%}")

            if is_fraud:
                st.error("🚨 Warning: This job posting looks suspicious!")
                st.markdown("""
                **Why it might be fake:**
                - Overly high salary for minimal requirements.
                - Vague company information.
                - Urgency or "Immediate Start" keywords.
                - Requests for payment or personal banking info.
                """)
            else:
                st.success("✅ This job posting appears to be legitimate.")
                st.markdown("""
                **Analysis:**
                - Professional language and structure detected.
                - Clear requirements and responsibilities.
                - Standard industry terminology used.
                """)

# Footnote
st.markdown("---")
st.caption("Developed by Jules - ML Engineer")
