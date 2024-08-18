from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings


embedding_function = OllamaEmbeddings(model="all-minilm")

 
def upsert_to_chromadb(docs):
    db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")


def initialize_chromadb():
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    return db