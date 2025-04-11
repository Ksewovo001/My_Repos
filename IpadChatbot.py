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
except:
    st.error("Failed to load Student Accounts dataset or embeddings.")
    st.stop()

# Load Admissions data
try:
    df_admissions = pd.read_csv('Admissions.csv')
    if 'Question' not in df_admissions.columns:
        raise ValueError("The CSV must contain a 'Question' column.")
    admissions_questions = df_admissions['Question'].dropna().astype(str).tolist()
    embeddings_admissions = np.array(model.encode(admissions_questions))
except Exception as e:
    st.error("Failed to load Admissions data or encode questions.")
    st.exception(e)
    st.stop()

# Mobile-friendly style with black label fix
st.markdown("""
    <style>
    .stApp {
        background-color: #f9f9f9;
        padding-bottom: 40px;
    }
    h1, h3, p {
        text-align: center;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
        padding: 12px;
    }
    .stButton>button {
        font-size: 18px;
        padding: 10px 20px;
        width: 100%;
    }
    .css-1cpxqw2 {
        overflow-y: auto;
    }
    /* Force label (prompt) text to black */
    label {
        color: black !important;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <h1 style='color: #d62828;'>ISU Parents Q&A Chatbot</h1>
    <p>Helping parents and families find answers, faster.</p>
""", unsafe_allow_html=True)

# Load image
try:
    image = Image.open("Chatbot.png")
    st.image(image, width=150)
except:
    pass

# Input interface
category = st.selectbox("Select a topic:", ["Student Accounts", "Admissions"])
st.markdown("<h3>Ask your question below:</h3>", unsafe_allow_html=True)
user_input = st.text_input("Your question", key="user_question")

# Query logic
def answer_query(question, data_df, embeddings):
    if not question.strip():
        return None, None, 0.0

    q_vec = np.array(model.encode(question))
    sims = embeddings.dot(q_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec))
    best_i = np.argmax(sims)

    answer_col = "Answer" if "Answer" in data_df.columns else "Answers"
    question_col = "Question" if "Question" in data_df.columns else "Questions"

    # Check for missing column
    if answer_col not in data_df.columns:
        return "Error: Answer column missing in the dataset.", "", 0.0

    return data_df.iloc[best_i][answer_col], data_df.iloc[best_i][question_col], sims[best_i]

# Run only if input exists and button is clicked
if user_input and st.button("Get Answer"):
    st.write("You asked:", user_input)

    if category == "Student Accounts":
        answer, matched_q, score = answer_query(user_input, df_accounts, embeddings_accounts)
    else:
        answer, matched_q, score = answer_query(user_input, df_admissions, embeddings_admissions)

    if answer and score > 0.3:
        st.subheader("Answer:")
        st.write(answer)
        st.caption(f"Matched Question: {matched_q}")
        st.caption(f"Similarity Score: {score:.3f}")
    elif answer and score <= 0.3:
        st.markdown(
    "<div style='background-color:#fff3cd; padding: 10px; border-radius:5px; color: black;'>"
    "<strong>Warning:</strong> No strong match found. Try rephrasing your question."
    "</div>",
    unsafe_allow_html=True
)
    else:
        st.warning("Please enter a valid question.")
