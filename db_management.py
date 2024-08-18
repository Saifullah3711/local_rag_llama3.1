import streamlit as st
import tempfile
import os
from pdf_parsing import read_and_parse_pdf
from upsert_chromadb import upsert_to_chromadb, initialize_chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings


embedding_function = OllamaEmbeddings(model="all-minilm")

def get_unique_files():
    unique_files = []
    db = initialize_chromadb()
    for x in range(len(db.get()["ids"])):
        doc = db.get()["metadatas"][x]
        source = doc["source"]
        if source not in unique_files:
            unique_files.append(source)
    return unique_files


def database_management():
    st.title("Database Management")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        st.write("Uploading file:", uploaded_file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_file_path = tmp_file.name
        
        docs = read_and_parse_pdf(temp_file_path)

        os.remove(temp_file_path)

        upsert_to_chromadb(docs)
        st.write("Documents upserted to ChromaDB")

    db = initialize_chromadb()
    unique_files = get_unique_files()
    selected_file = st.multiselect("Select a file", unique_files)
    if selected_file:
        st.write("Selected files are: ",selected_file)

        if st.button("Delete selected files"):
            for file in selected_file:
                rem_ids = db.get(where={"source": file})['ids']
                for id in rem_ids:
                    db.delete(ids=id)
  

