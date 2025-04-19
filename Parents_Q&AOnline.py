import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from PIL import Image

# Set up Streamlit page - keeping it simple
st.set_page_config(page_title="ISU Parents Chatbot", layout="centered")

# How I load the embedding model
try:
    embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
except Exception as err:
    st.error("Ugh... couldn't load the model. Something went wrong.")
    st.exception(err)
    st.stop()

# To get dataset and other related files (embeddings) ready
try:
    big_df = pd.read_csv("Big_Dataset.csv")  # Big_Dataset.csv not "Big Dataset.csv"
    
    with open("Big_Dataset_embeddings.pkl", "rb") as pkl_file:
        embedded_data = np.array(pickle.load(pkl_file))

    # To normalize embeddings for the similarity score (actual formulacoming down)
    embedded_data = embedded_data / np.linalg.norm(embedded_data, axis=1, keepdims=True)

except Exception as e:
    st.error("Hmm... either the dataset or the embeddings didn’t load properly.")
    st.exception(e)
    st.stop()

# Just some styling for aesthetic
st.markdown("""
    <style>
        .stApp { background-color: #f9f9f9; }
        h1, h3, p { text-align: center; }
        .question-input input {
            border-radius: 8px;
            padding: 10px;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and the intro
st.markdown("""
    <h1 style='color: #d62828;'>🎓 ISU Parents Q&A Chatbot 🧾</h1>
    <p>Helping parents and families with quick and reliable answers.</p>
""", unsafe_allow_html=True)

# Adding some fun images for the context
img_cols = st.columns([1, 1, 1])  # Going for a balanced layout

try:
    with img_cols[0]:
        img_left = Image.open("Dr_Birdiclopedia2.png")
        st.image(img_left, width=200)

    with img_cols[2]:
        img_right = Image.open("Dr_Birdiclopedia1.png")
        st.image(img_right, width=200)
except Exception as img_err:
    st.error("Couldn’t load one or both images... might be missing?")
    st.exception(img_err)

# Greeting from the chatbot
st.markdown("""
    <h3>Hello! I'm Dr. Birdiclopedia. Got questions for me? Ask away!!! 👇</h3>
""", unsafe_allow_html=True)

# Some examples prompts to guide first users
st.markdown("""
<p style='text-align: center;'><strong>Try asking:</strong></p>
<ul style='text-align: center; list-style-type: none; padding-left: 0;'>
    <li><em>"What is the Redbird Scholarship Program?"</em></li>
    <li><em>"How do I apply for housing?"</em></li>
    <li><em>"What types of scholarships are offered?"</em></li>
</ul>
""", unsafe_allow_html=True)

# Little dropdown prompt to explain
with st.expander("❓ How it works"):
    st.info("Type your question and click Submit. I’ll search for the best match and get back to you with a helpful answer!")

# For the actual question form
with st.form("question_form"):
    user_question = st.text_input("Enter your question:", key="user_q")
    was_submitted = st.form_submit_button("Submit")

# A bit of a comment for the query function (In case I might forget how this works later)
def find_best_answer(query_text, df_ref, vector_data):
    if not query_text.strip():
        return None, None, 0.0  # Just return nothing if user did not enter a question
    
    # To create embedding for the input
    try:
        query_vec = embed_model.encode(query_text, normalize_embeddings=True)
    except Exception as enc_err:
        print("Encoding issue:", enc_err)
        return None, None, 0.0

    # Formula fro the Cosine similarity
    sim_scores = vector_data.dot(query_vec)
    top_index = np.argmax(sim_scores)

    # To handle different CSV column names (Just in case they're inconsistent)
    answer_field = "Answer" if "Answer" in df_ref.columns else "Answers"
    question_field = "Question" if "Question" in df_ref.columns else "Questions"

    return df_ref.iloc[top_index][answer_field], df_ref.iloc[top_index][question_field], sim_scores[top_index]

# When the user hits "Submit"
if was_submitted:
    if not user_question.strip():
        st.warning("Oops! You need to type something first.")
    else:
        best_ans, matched_q_text, similarity = find_best_answer(user_question, big_df, embedded_data)

        if best_ans and similarity > 0.3:
            st.subheader("Here is what I found for you:")
            st.write(best_ans)
            st.caption(f"Matched Question: {matched_q_text}")
            st.caption(f"Similarity Score: {similarity:.3f}")
        else:
            st.subheader("Answer:")
            st.write("While I'm working hard to provide you with an accurate answer, Try checking the [Parents and Family Resources](https://parents.illinoisstate.edu/) page just in case.")

# Extra info link at the bottom
st.markdown("""
<div style='text-align: center; margin-top: 10px;'>
    Need more info? Visit the <a href='https://parents.illinoisstate.edu/' target='_blank'><strong>Parents & Family Services</strong></a> page.
</div>
""", unsafe_allow_html=True)

# For my team mates who helped make this project work.
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
