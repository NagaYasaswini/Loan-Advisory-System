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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

# Initialize embeddings and LLM
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.7)
except Exception as e:
    st.error(f"Failed to initialize Gemini API: {str(e)}")
    sys.exit(1)

# Load and index knowledge base from multiple files
def load_knowledge_base(folder_path):
    try:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Knowledge-base folder '{folder_path}' not found.")
        
        documents = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_name.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif file_name.endswith('.csv'):
                loader = CSVLoader(file_path)
                documents.extend(loader.load())
            else:
                st.warning(f"Skipping unsupported file: {file_name}")

        if not documents:
            raise ValueError("No valid documents found in the knowledge-base folder.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
        vectorstore.persist()
        return vectorstore
    except Exception as e:
        st.error(f"Error loading knowledge-base: {str(e)}")
        return None

# RAG-based policy retrieval
def get_policy_response(query, vectorstore):
    try:
        if not vectorstore:
            raise ValueError("Vector store is not initialized.")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
        )
        return qa_chain.run(query)
    except Exception as e:
        st.error(f"Error processing policy query: {str(e)}")
        return "Unable to retrieve policy information at this time."

# OCR for document processing
def extract_text_from_image(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError("Uploaded file not found.")
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        if not text.strip():
            raise ValueError("No text detected in the uploaded image.")
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return None

# Eligibility insight from document
def get_eligibility_insight(text, vectorstore):
    try:
        if not text:
            raise ValueError("No text provided for eligibility analysis.")
        prompt = f"Analyze this document text for loan eligibility using relevant policies: {text}. Provide insights on income, employment, compliance, and best-fit loans."
        return get_policy_response(prompt, vectorstore)  # Use RAG for contextual analysis
    except Exception as e:
        st.error(f"Error analyzing eligibility: {str(e)}")
        return "Eligibility analysis unavailable."

# Parse and analyze applicant data
def parse_applicant_data(input_str):
    try:
        if not input_str or ':' not in input_str:
            raise ValueError("Invalid applicant data format. Use 'key:value, key:value' (e.g., 'income:50000, credit_score:700').")
        data = dict(item.strip().split(':') for item in input_str.split(','))
        return pd.DataFrame([data])
    except Exception as e:
        st.error(f"Error parsing applicant data: {str(e)}")
        return None

def analyze_applicant_data(df, vectorstore):
    try:
        if df is None or df.empty:
            raise ValueError("No valid applicant data to analyze.")
        data_str = df.to_string()
        approval_prompt = f"Based on applicant data {data_str} and loan policies from the knowledge-base, predict approval chance (high/medium/low) with reasoning."
        approval = llm.predict(approval_prompt)
        
        rec_prompt = f"Recommend best-fit loan products for applicant data {data_str}, considering policies, interest rates, and compliance."
        recommendation = get_policy_response(rec_prompt, vectorstore)
        return approval, recommendation
    except Exception as e:
        st.error(f"Error analyzing applicant data: {str(e)}")
        return "Unable to determine approval chance", "No loan recommendation available"

# Streamlit UI
st.title("AI-Powered Loan Advisory System")

# Load knowledge base once
if 'vectorstore' not in st.session_state:
    vectorstore = load_knowledge_base("knowledge-base")
    if vectorstore:
        st.session_state.vectorstore = vectorstore
    else:
        st.stop()  # Halt if knowledge base fails

# Policy Query Section
st.header("Loan Policy Inquiry")
policy_query = st.text_input("Ask about loan policies, interest rates, or compliance rules:")
if policy_query:
    response = get_policy_response(policy_query, st.session_state.vectorstore)
    st.write("Response:", response)

# Document Upload Section
st.header("Upload Documents for Eligibility (Payslips or Bank Statements)")
uploaded_file = st.file_uploader("Upload image or PDF", type=["png", "jpg", "pdf"])
if uploaded_file:
    try:
        temp_path = "temp_upload"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Basic PDF handling (extract first page as image for OCR; extend for multi-page if needed)
        if uploaded_file.type == "application/pdf":
            from pdf2image import convert_from_path
            images = convert_from_path(temp_path)
            text = pytesseract.image_to_string(images[0]) if images else ""
        else:
            text = extract_text_from_image(temp_path)
        
        if text:
            eligibility = get_eligibility_insight(text, st.session_state.vectorstore)
            st.write("Extracted Text:", text)
            st.write("Eligibility Insight:", eligibility)
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Applicant Data Input Section
st.header("Applicant Data Analysis")
applicant_input = st.text_input("Enter applicant data (e.g., income:50000, credit_score:700, age:30):")
if applicant_input:
    df = parse_applicant_data(applicant_input)
    if df is not None:
        approval, recommendation = analyze_applicant_data(df, st.session_state.vectorstore)
        st.write("Approval Chance:", approval)
        st.write("Recommended Loan:", recommendation)

# Run the app
if __name__ == "__main__":
    st.run()