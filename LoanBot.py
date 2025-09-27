# Imports ----------------------------------

import os
import pytesseract
from PIL import Image
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import (HuggingFaceEmbeddings, JinaEmbeddings)
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


# Load environment variables and Load API keys ------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables. Please check your .env file."
    )


# Streamlit UI Setup ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🏦", layout="centered")
st.title("💬 RAG Chatbot with ChromaDB")
st.header("LoanBot - Learn about Bank Loan Policy")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []



# Load Data sets --------------------------------
data = 'Knowledge-base'

all_doc= []

for filename in os.listdir(data):
    filepath=os.join(data, filename)

    if filename.lower().endswith(".txt"):
        loader = TextLoader(filepath)
    elif filename.lower().endswith(".csv"):
        loader = CSVLoader(filepath)
    elif filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    else:
        print(f"Skipping unsupported file: {filename}")
        continue
    docs = loader.load()
    all_doc.extend(docs)

print(f"Loaded {len(all_doc)} documents from {folder_path}")