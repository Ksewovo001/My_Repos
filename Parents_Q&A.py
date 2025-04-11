import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from PIL import Image

st.set_page_config(page_title="ISU Parents Chatbot", layout="centered")

# Load model
try:
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
except Exception as e:
    st.error("Error loading the model.")
    st.exception(e)
    st.stop()

# Load Student Accounts data
try:
    df_accounts = pd.read_csv('Student_Accounts_Embedded.csv')
    with open('question_embeddings.pkl', 'rb') as f:
        embeddings_accounts = np.array(pickle.load(f))
    # Normalize once if not already
    embeddings_accounts = embeddings_accounts / np.linalg.norm(embeddings_accounts, axis=1, keepdims=True)
except Exception as e:
    st.error("Failed to load Student Accounts dataset or embeddings.")
    st.exception(e)
    st.stop()

# Load Admissions data
try:
    df_admissions = pd.read_csv('Admissions.csv')
    if 'Question' not in df_admissions.columns:
        raise ValueError("The CSV must contain a 'Question' column.")
    admissions_questions = df_admissions['Question'].dropna().astype(str).tolist()
    embeddings_admissions = model.encode(admissions_questions, normalize_embeddings=True)
except Exception as e:
    st.error("Failed to load Admissions data or encode questions.")
    st.exception(e)
    st.stop()

# Custom CSS
st.markdown("""
    <style>
        .stApp { background-color: #f9f9f9; }
        h1, h3, p { text-align: center; }
        .question-input input {
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <h1 style='color: #d62828;'>🎓 ISU Parents Q&A Chatbot 🧾</h1>
    <p>Helping parents and families find answers, faster.</p>
""", unsafe_allow_html=True)

# Image
try:
    image = Image.open("Chatbot.png")
    st.image(image, width=180)
except:
    pass

# Topic and input form
category = st.selectbox("Select a topic:", ["Student Accounts", "Admissions"])
st.markdown("<h3>Ask your question below 👇</h3>", unsafe_allow_html=True)

with st.form("question_form"):
    user_input = st.text_input("Enter your question:", key="user_question")
    submitted = st.form_submit_button("Submit")

# Query logic
def answer_query(question, data_df, embeddings):
    if not question.strip():
        return None, None, 0.0
    # Normalize question vector
    q_vec = model.encode(question, normalize_embeddings=True)
    sims = embeddings.dot(q_vec)
    best_i = np.argmax(sims)
    answer_col = "Answer" if "Answer" in data_df.columns else "Answers"
    question_col = "Question" if "Question" in data_df.columns else "Questions"
    return data_df.iloc[best_i][answer_col], data_df.iloc[best_i][question_col], sims[best_i]

# Handle query
if submitted and user_input:
    if category == "Student Accounts":
        answer, matched_q, score = answer_query(user_input, df_accounts, embeddings_accounts)
    else:
        answer, matched_q, score = answer_query(user_input, df_admissions, embeddings_admissions)

    if answer and score > 0.3:
        st.subheader("Answer:")
        st.write(answer)
        st.caption(f"Matched Question: {matched_q}")
        st.caption(f"Similarity Score: {score:.3f}")
    else:
        st.warning("No strong match found. Try rephrasing your question.")
