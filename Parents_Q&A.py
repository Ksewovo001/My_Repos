import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv('C:\\Users\\adelp\\OneDrive\\Documents\\BIS 362\\Student_Accounts_Embedded.csv')
with open('question_embeddings.pkl', 'rb') as f:
    question_embeddings = pickle.load(f)

print(f"Loaded {len(df)} Q&A pairs.")
print(f"Number of embedding vectors: {len(question_embeddings)}")

model = SentenceTransformer('all-MiniLM-L6-v2')

user_question = "Can I pay my bill with a credit card?"
query_vec = np.array(model.encode(user_question))
question_embeddings = np.array(question_embeddings)

dot_products = question_embeddings.dot(query_vec)
norms = np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(query_vec)
cosine_similarities = dot_products / norms

best_idx = np.argmax(cosine_similarities)
best_score = cosine_similarities[best_idx]

best_question = df.iloc[best_idx]["Questions"]
best_answer = df.iloc[best_idx]["Answers"]

print(f"\nUser's question: {user_question}")
print(f"Best match from FAQ: {best_question}")
print(f"Answer: {best_answer} (Score: {best_score:.3f})")

def answer_query(question):
    q_vec = np.array(model.encode(question))
    dots = question_embeddings.dot(q_vec)
    sims = dots / (np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(q_vec))
    best_i = np.argmax(sims)
    return df.iloc[best_i]["Answers"], df.iloc[best_i]["Questions"], sims[best_i]


st.title("Student Account Chatbot")
st.write("Ask me a question about your student account.")

user_input = st.text_input("Enter your question:")

if user_input:
    answer, matched_q, score = answer_query(user_input)
    st.subheader("Answer:")
    st.write(answer)
    st.caption(f"Matched Question: {matched_q}")
    st.caption(f"Similarity Score: {score:.3f}")
