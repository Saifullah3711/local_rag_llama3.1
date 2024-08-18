from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



def read_and_parse_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)

    pdf_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    
    docs = text_splitter.split_documents(pdf_docs)
    return docs