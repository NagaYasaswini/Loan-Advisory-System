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
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------- LOAD ENV ------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# ---------------- STREAMLIT SETUP -----------------
st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")
st.title("💬 RAG Chatbot with OCR & Image Captioning")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR OPTIONS -----------------
st.sidebar.header("Options")
task_option = st.sidebar.radio("Choose task:", ["OCR", "Image Captioning"])
uploaded_file = st.sidebar.file_uploader(
    f"Upload a file for {task_option}", type=["png", "jpg", "jpeg", "pdf"]
)

# ---------------- OCR / IMAGE CAPTION -----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ocr_text = ""
img_caption = ""
if uploaded_file:
    if task_option == "OCR":
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(uploaded_file)
            pages = loader.load_and_split()
            ocr_text = "\n".join([page.page_content for page in pages])
        else:
            image = Image.open(uploaded_file).convert("RGB")
            ocr_text = pytesseract.image_to_string(image)

        with st.sidebar.expander("Preview Extracted Text"):
            st.text_area("OCR Text", ocr_text, height=200)

    else:  # Image Captioning
        image = Image.open(uploaded_file).convert("RGB")
        st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Generating image caption..."):
            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            inputs = processor(images=image, return_tensors="pt")
            out = model.generate(**inputs)
            img_caption = processor.decode(out[0], skip_special_tokens=True)

        with st.sidebar.expander("Image Caption"):
            st.write(img_caption)

# ---------------- LOAD DOCUMENTS & VECTORSTORE -----------------
with open(
    "Loan-Advisory-System/Knowledge-base/Policy_doc.md", "r", encoding="utf-8"
) as f:
    markdown_document = f.read()

markdown_splitter = MarkdownHeaderTextSplitter([("#", "Header 1")], strip_headers=False)
md_splits = markdown_splitter.split_text(markdown_document)

embeddings = JinaEmbeddings(jina_api_key=JINA_API_KEY, model_name="jina-embeddings-v3")
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest", google_api_key=GEMINI_API_KEY, temperature=0.7
)

vectorstore = Chroma.from_documents(
    md_splits, embeddings, persist_directory="./chroma_db"
)
vectorstore.persist()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

# ---------------- CHAT UI -----------------
st.subheader("💬 Ask Questions")
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:", "")
    submit = st.form_submit_button("Send")

if submit and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Decide source: OCR text / Image Caption / RAG knowledge base
    if uploaded_file:
        if task_option == "OCR" and ocr_text:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            chunks = splitter.split_text(ocr_text)
            temp_store = Chroma.from_documents(
                chunks, embeddings, persist_directory=None
            )
            temp_chain = RetrievalQA.from_chain_type(
                llm=llm, retriever=temp_store.as_retriever(search_kwargs={"k": 5})
            )
            with st.spinner("Processing OCR document..."):
                result = temp_chain.run(user_input)

        elif task_option == "Image Captioning" and img_caption:
            result = f"Image Caption: {img_caption}\n\nUser Query: {user_input}"

        else:
            with st.spinner("Thinking..."):
                result = qa_chain.run(user_input)
    else:
        with st.spinner("Thinking..."):
            result = qa_chain.run(user_input)

    st.session_state.messages.append({"role": "assistant", "content": result})

# Display chat history nicely
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")
