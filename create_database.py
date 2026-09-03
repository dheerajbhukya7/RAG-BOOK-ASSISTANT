from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


# =========================
# 1. Load PDF
# =========================

pdf_path = r"M:\Machine Learning\RAG\Documentloaders\Deeplearning.pdf"

loader = PyPDFLoader(pdf_path)

docs = loader.load()

print(f"Loaded {len(docs)} pages")


# =========================
# 2. Clean PDF text
# =========================

def clean_text(text):
    # Remove invalid Unicode characters
    text = text.encode("utf-8", "ignore").decode("utf-8")

    # Normalize whitespace
    text = " ".join(text.split())

    return text


for doc in docs:
    doc.page_content = clean_text(doc.page_content)


print("PDF text cleaned")


# =========================
# 3. Split into chunks
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400
)

splits = text_splitter.split_documents(docs)

print(f"Created {len(splits)} chunks")


# =========================
# 4. Create embeddings
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

print("Embedding model loaded")


# =========================
# 5. Test chunks
# =========================

texts = [doc.page_content for doc in splits]

for i, text in enumerate(texts):
    try:
        embeddings.embed_documents([text])
    except Exception as e:
        print(f"❌ Bad chunk: {i}")
        print(repr(text[:500]))
        print(e)
        raise

print("✅ All chunks embedded successfully")


# =========================
# 6. Store in Chroma
# =========================

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ Chroma database created successfully!")