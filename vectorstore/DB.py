from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
data = PyPDFLoader(r"M:\Machine Learning\RAG\Documentloaders\genai-principles.pdf")
docs = data.load()

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Extract texts and metadatas separately - this avoids the batching issue
texts = [doc.page_content for doc in splits]
metadatas = [doc.metadata for doc in splits]

# Using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Use from_texts instead of from_documents to avoid the batching bug
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="chroma_db"
)
results = vectorstore.similarity_search("What are the principles of generative AI?", k=3)

for r in results:
    print(r.page_content)
    print()

retriver = vectorstore.as_retriever()

