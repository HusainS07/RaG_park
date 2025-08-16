import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Load PDF
loader = PyPDFLoader("sodapdf-converted.pdf")
documents = loader.load()

# Split into chunks (reduce number of chunks if memory issues)
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
chunks = splitter.split_documents(documents)

# Embed and persist
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="./chroma_db")
vectorstore.persist()

print("Embeddings persisted successfully!")
