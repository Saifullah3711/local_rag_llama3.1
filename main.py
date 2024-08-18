import streamlit as st
from chatbot import chatbot_chat
from db_management import database_management


if __name__ == "__main__":
    st.set_page_config(page_title="RAG Chatbot", page_icon="🔗")
    st.sidebar.title("Llama3.1 RAG Chatbot")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Chatbot", "Database Management"]
    )
    if app_mode == "Chatbot":
        chatbot_chat()
    elif app_mode == "Database Management":
        database_management()