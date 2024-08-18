from langchain_ollama import OllamaLLM
import streamlit as st
from upsert_chromadb import initialize_chromadb
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model = OllamaLLM(model = "llama3.1")
db = initialize_chromadb()
retriever = db.as_retriever()
prompt_llm = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def chatbot_chat():
    st.title("Chat with PDF - Llama3.1")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_llm
    | model
    | StrOutputParser()
)


    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = rag_chain.stream(prompt)
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
