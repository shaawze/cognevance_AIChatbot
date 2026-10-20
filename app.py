
import streamlit as st
import joblib
import json
import random

st.set_page_config(
    page_title="NLP Chatbot",
    page_icon="🤖"
)

@st.cache_resource
def load_model():
    return joblib.load("chatbot_model.pkl")

model = load_model()

with open("responses.json", "r") as file:
    responses = json.load(file)

st.title("🤖 NLP Chatbot")
st.write("Your friendly AI chatbot built with Python and NLP!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    predicted_intent = model.predict([user_input])[0]
    reply = random.choice(responses[predicted_intent])

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })

    st.rerun()
