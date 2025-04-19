import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from PIL import Image

st.set_page_config(page_title="ISU Parents Chatbot", layout="centered")

try:
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
except Exception as e:
    st.error("Error loading the model.")
    st.exception(e)
    st.stop()

try:
    df_big = pd.read_csv('Big_Dataset.csv')
    with open('Big_Dataset_embeddings.pkl', 'rb') as f:
        embeddings_big = np.array(pickle.load(f))
    embeddings_big = embeddings_big / np.linalg.norm(embeddings_big, axis=1, keepdims=True)
except Exception as e:
    st.error("Failed to load Big Dataset or embeddings.")
    st.exception(e)
    st.stop()

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

st.markdown("""
    <h1 style='color: #d62828;'>🎓 ISU Parents Q&A Chatbot 🧾</h1>
    <p>Helping parents and families find answers, faster.</p>
""", unsafe_allow_html=True)

cols = st.columns([0.5, 1, 0.5, 1])

try:
    with cols[1]:
        image1 = Image.open("Dr_Birdiclopedia2.png")
        st.image(image1, width=200)
    with cols[3]:
        image2 = Image.open("Dr_Birdiclopedia1.png")
        st.image(image2, width=200)
except Exception as e:
    st.error("One or more images could not be loaded.")
    st.exception(e)


    

st.markdown("""
    <h3>Hello! I'm Dr. Birdiclopedia. Got questions for me? Ask away!!! 👇</h3>
""", unsafe_allow_html=True)

with st.form("question_form"):
    user_input = st.text_input("Enter your question:", key="user_question")
    submitted = st.form_submit_button("Submit")

def answer_query(question, data_df, embeddings):
    if not question.strip():
        return None, None, 0.0
    q_vec = model.encode(question, normalize_embeddings=True)
    sims = embeddings.dot(q_vec)
    best_i = np.argmax(sims)
    answer_col = "Answer" if "Answer" in data_df.columns else "Answers"
    question_col = "Question" if "Question" in data_df.columns else "Questions"
    return data_df.iloc[best_i][answer_col], data_df.iloc[best_i][question_col], sims[best_i]

if submitted and user_input:
    answer, matched_q, score = answer_query(user_input, df_big, embeddings_big)

    if answer and score > 0.3:
        st.subheader("Here is what I found for you:")
        st.write(answer)
        st.caption(f"Matched Question: {matched_q}")
        st.caption(f"Similarity Score: {score:.3f}")
    else:
        st.subheader("Answer:")
        st.write("While I'm working hard to provide you with an accurate answer, please refer to the [Parents and Family Resources](https://studentaccess.illinoisstate.edu/parents/) page for more details.")

st.markdown("""
    <hr style='margin-top: 2em;'>
    <div style='text-align: center; font-size: 14px; line-height: 1.8;'>
        <div><strong>Created by</strong></div>
        <div style='margin-top: 5px;'>
            Kankoe S. &nbsp;&nbsp;&nbsp; Rex A. &nbsp;&nbsp;&nbsp; Alaa H. &nbsp;&nbsp;&nbsp; Tyler C. &nbsp;&nbsp;&nbsp; Nathan P.
        </div>
        <div style='margin-top: 5px;'>Aidan D.</div>
    </div>
""", unsafe_allow_html=True)

