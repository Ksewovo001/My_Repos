import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from PIL import Image


model = SentenceTransformer("Ksewovo001/My-Repos-Model")


df_accounts = pd.read_csv('Student_Accounts_Embedded.csv')
with open('question_embeddings.pkl', 'rb') as f:
    embeddings_accounts = np.array(pickle.load(f))

df_admissions = pd.read_csv('Admissions.csv')
admissions_questions = df_admissions['Question'].tolist()
embeddings_admissions = np.array(model.encode(admissions_questions))


st.markdown(
    """
    <style>
        .stApp {
            background-color: #f9f9f9;
        }
        h1, h3, p {
            text-align: center;
        }
        .question-input input {
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("""
    <h1 style='color: #d62828;'>🎓 ISU Parents Q&A Chatbot</h1>
    <p>Helping parents and families find answers, faster.</p>
""", unsafe_allow_html=True)


try:
    image = Image.open("Chatbot.png")
    st.image(image, width=180)
except:
    pass


category = st.selectbox("Select a topic:", ["Student Accounts", "Admissions"])


st.markdown("<h3>Ask your question below 👇</h3>", unsafe_allow_html=True)
user_input = st.text_input("Enter your question:", key="user_question")


def answer_query(question, data_df, embeddings):
    q_vec = np.array(model.encode(question))
    sims = embeddings.dot(q_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec))
    best_i = np.argmax(sims)
    answer_col = "Answer" if "Answer" in data_df.columns else "Answers"
    question_col = "Question" if "Question" in data_df.columns else "Questions"
    return data_df.iloc[best_i][answer_col], data_df.iloc[best_i][question_col], sims[best_i]


if user_input:
    if category == "Student Accounts":
        answer, matched_q, score = answer_query(user_input, df_accounts, embeddings_accounts)
    else:
        answer, matched_q, score = answer_query(user_input, df_admissions, embeddings_admissions)

    st.subheader("Answer:")
    st.write(answer)
    st.caption(f"Matched Question: {matched_q}")
    st.caption(f"Similarity Score: {score:.3f}")
