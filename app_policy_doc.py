# IMPORTS -----------------------------------------
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import sys
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
import pytesseract
from PIL import Image
from langchain_community.embeddings import JinaEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
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

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")
st.title("💬 RAG Chatbot with ChromaDB")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# LOAD DOCS -----------------------------------------------------------------
with open(
    "Loan-Advisory-System/Knowledge-base/Policy_doc.md", "r", encoding="utf-8"
) as file:
    markdown_document = file.read()


# CHUNK the markdown document USING LANGCHAIN ------------------------------
with open(
    "Loan-Advisory-System/Knowledge-base/Policy_doc.md", "r", encoding="utf-8"
) as file:
    markdown_document = file.read()

headers_to_split_on = [
    ("#", "Header 1"),
    # ("##", "Header 2"),
    # ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
md_header_splits = markdown_splitter.split_text(markdown_document)

# JINA EMBEDDINGS MODEL INITIALIZATION ANS LOAD CHAT MODEL LLM---------------------------------------------------------------
embeddings = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_API_KEY"), model_name="jina-embeddings-v3"
)
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest", google_api_key=GEMINI_API_KEY, temperature=0.7
)


# DEFINING THE CHROMA DB VECTOR STORE ----------------------------------------------------------------
vectorstore = Chroma.from_documents(
    md_header_splits, embeddings, persist_directory="./chroma_db"
)
vectorstore.persist()


# QUERYING GEMINI LLM VIA RAG QA CHAIN --------------------------------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)
query = "What are the eligibility criteria in the Loan Advisory Policy?"
response = qa_chain.invoke({"query": query})
print(response["result"])

# LOADING STREAMLIT UI ---------------------------------------------------------------------------------------

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask me anything:", "")
    submit = st.form_submit_button("Send")

if submit and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get response from RAG QA
    with st.spinner("Thinking..."):
        result = qa_chain.run(user_input)  # changes qa_chain

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result})


for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")


# Run the app
# if __name__ == "__main__":
#     st.run()
