# Loan Advisory System  

This project is an AI-powered Loan Advisory application designed to answer queries related to **loan applications** and **RBI policy documents**. It leverages document embedding, vector databases, and LLMs to create a searchable knowledge system.  

---

## 📌 Project Overview  

- Collected data from:  
  - **Hugging Face Dataset** → Loan Prediction dataset.  
  - **RBI Official Website** → RBI policy documents related to loans.  

- **Data Processing**:  
  - Combined multiple data sources (**CSV + PDF**) into a single **Markdown file (MD)**.  
  - Chunked the dataset into manageable pieces for efficient retrieval.  

- **Vector Database**:  
  - Stored processed data in **ChromaDB**.  

- **Embeddings & LLM**:  
  - Used **local Hugging Face embeddings** to embed the text data.  
  - Integrated **Gemini API** as the LLM for query answering.  

- **Deployment**:  
  - Deployed the application using **Streamlit** for an interactive user interface.  

---

## ⚙️ Tech Stack  

- **Python**  
- **LangChain**  
- **Hugging Face Embeddings**  
- **ChromaDB**  
- **Gemini API (LLM)**  
- **Streamlit**  

---

## 🚀 How It Works  

1. Data is collected from Hugging Face loan dataset & RBI website.  
2. Data sources (CSV, PDF) are combined and converted into Markdown format.  
3. Text data is split into smaller chunks.  
4. Chunks are embedded using **Hugging Face embeddings**.  
5. Embeddings are stored in **ChromaDB**.  
6. Queries from the user are processed by **Gemini API** LLM with context retrieved from ChromaDB.  
7. The Streamlit app displays relevant answers related to loan applications and RBI policies.  

---

## 🖥️ Running the Project  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/loan-advisory-system.git
   cd loan-advisory-system
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser and start querying loan-related information.  

---

## 📚 Use Case  

- Helps users get quick answers about **loan applications**.  
- Provides insights from **RBI policies**.  
- Reduces manual search effort by making loan-related data **searchable and conversational**.  

