import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Setup Streamlit UI
st.set_page_config(page_title="PDF RAG Q&A", layout="wide")
st.title("📄 PDF RAG Application")

# Sidebar for file upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload your document", type="pdf")

# Global variables/State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Logic Functions ---

def clean_text(text):
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = " ".join(text.split())
    return text

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    # Create new index for each upload
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db_temp" # Use temporary dir for session
    )
    return vectorstore

# --- Application Flow ---

if uploaded_file:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.session_state.vectorstore = process_pdf(tmp_path)
        os.remove(tmp_path)
        st.success("PDF processed and indexed!")

# Chat Interface
if st.session_state.vectorstore:
    st.header("Chat with your PDF")

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Retrieval
            retriever = st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
            docs = retriever.invoke(prompt)
            context = "\n".join([doc.page_content for doc in docs])

            # LLM Response
            llm = ChatMistralAI(model_name="open-mistral-7b", temperature=0.1)
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use context to answer. If you don't know, say you don't know."),
                ("human", "Context: {context}\nQuestion: {question}")
            ])

            final_prompt = chat_prompt.invoke({'context': context, 'question': prompt})
            response = llm.invoke(final_prompt)

            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
else:
    st.info("Please upload a PDF file in the sidebar to start asking questions.")
