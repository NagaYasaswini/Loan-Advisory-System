# ------------- Importing ------------------
import os
import pytesseract
from PIL import Image
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.embeddings import (HuggingFaceEmbeddings, JinaEmbeddings)
from langchain_google_genai import ChatGoogleGenerativeAI
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



# Load Markdown data
with open("Data/RBI-Policies.md", "r", encoding="utf-8") as file:
    markdown_document = file.read() 

# -------------- Reading PDF -----------
pdf_read=PyPDFLoader('Data/Co-origination of loans by Banks and NBFCs for lending to priority sector.pdf').load()
csv_read = CSVLoader('Data/loan_applications.csv').load()

# ----------- Combining the documents from csv into a text -----------------
text1 = '\n\n'.join([x.page_content for x in pdf_read])
text2 = '\n\n'.join([y.page_content for y in csv_read])

all_data = f"Combined data of PDF, CSV\n\n"

all_data += f"Markdown file\n\n{markdown_document}\n\n"
all_data += f"## PDF Content\n\n{text1}\n\n"
all_data += f"## CSV Content\n\n{text2}\n\n"

# ------------ Create a file with all the data ------------
with open('Data/all_data.md', 'w', encoding='utf-8') as f:
    f.write(all_data)



# Chunk the data ------------------------
headers_to_split_on = [
    ("#", "Header 1"),
    # ("##", "Header 2"),
    # ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers = False)
md_header_splits = markdown_splitter.split_text(all_data)

#sentences = [doc.page_content for doc in md_header_splits]



# ---------------------- Embeddings with HuggingFace----------------

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



# ---------- LLM Model ------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest", google_api_key=GEMINI_API_KEY, temperature=0.7
)



# DEFINING THE CHROMA DB VECTOR STORE ----------------------------------------------------------------
vectorstore = Chroma.from_documents(
    md_header_splits, embeddings, persist_directory="./chroma-db"
)
vectorstore.persist()

# QUERYING GEMINI LLM VIA RAG QA CHAIN --------------------------------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)
query = "What are the eligibility criteria in the Loan Advisory Policy?"
response = qa_chain.invoke({"query": query})
print(response["result"])


# Streamlit UI Setup ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🏦", layout="centered")
st.title("💬 RAG Chatbot with ChromaDB")
st.header("LoanBot - Learn about Bank Loan Policy")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []



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
