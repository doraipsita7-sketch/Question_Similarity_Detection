import streamlit as st
import joblib
import os
import numpy as np

from scipy.sparse import hstack

st.set_page_config(
    page_title="Question Similarity AI",
    page_icon="🔍",
    layout="wide"
)

MODEL_PATH = "models/question_similarity_model.pkl"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #172554);
    color: #f8fafc;
}
.main .block-container {
    max-width: 1050px;
    padding-top: 2rem;
}
.hero {
    text-align: center;
    padding: 30px 18px;
    background: linear-gradient(135deg, #1e3a8a, #312e81);
    border-radius: 20px;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 36px;
}
.hero p {
    color: #dbeafe;
}
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-weight: bold;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.markdown("""
<div class="hero">
    <h1>🔍 Question Similarity AI</h1>
    <p>Intelligent Question Comparison using NLP & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.subheader("📝 Compare Your Questions")

col1, col2 = st.columns(2)

with col1:
    question1 = st.text_area(
        "Question 1",
        placeholder="What is the capital of India?",
        height=150
    )

with col2:
    question2 = st.text_area(
        "Question 2",
        placeholder="What is India's capital city?",
        height=150
    )


if st.button("🔎 Analyze Similarity", type="primary"):

    if not question1.strip() or not question2.strip():
        st.warning("Please enter both questions.")

    elif not os.path.exists(MODEL_PATH):
        st.error("Model not found. Run train_model.py first.")

    else:
        with st.spinner("Analyzing question pair..."):
            saved = load_model()
            vectorizer = saved["vectorizer"]
            model = saved["model"]

            q1 = vectorizer.transform([question1.strip()])
            q2 = vectorizer.transform([question2.strip()])

            difference = abs(q1 - q2)
            shared = q1.multiply(q2)

            features = hstack(
                [difference, shared],
                format="csr",
                dtype=np.float32
            )

            prediction = int(model.predict(features)[0])
            probabilities = model.predict_proba(features)[0]
            confidence = float(probabilities[prediction]) * 100

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success("✅ Questions are Similar")
        else:
            st.error("❌ Questions are Not Similar")

        m1, m2 = st.columns(2)

        with m1:
            st.metric("Model Confidence", f"{confidence:.2f}%")

        with m2:
            st.metric(
                "Predicted Class",
                "Similar" if prediction == 1 else "Not Similar"
            )

        st.progress(confidence / 100)

        st.info(
            "Model confidence is an estimated probability, "
            "not a direct semantic similarity percentage."
        )

st.divider()

m1, m2, m3 = st.columns(3)
m1.metric("NLP Technique", "TF-IDF")
m2.metric("ML Algorithm", "Logistic Regression")
m3.metric("Test Accuracy", "79.53%")

st.caption(
    "Question Similarity Detection System | Python • Streamlit • NLP • ML"
)