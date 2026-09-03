from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 10,
        "lambda_mult": 0.5
        }

)

# Updated: Using open-mistral-7b which is available via the Mistral Cloud API
llm = ChatMistralAI(model_name="open-mistral-7b", temperature=0.1)

#prompttemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful assistant that answers questions based on the context provided.
If you don't know the answer,
just say that you don't know, don't try to make up an answer."""),
        ("human","""context: {context}
         Question: {question}""")
])

print("Rag system initialized successfully")

print("press 0 to exit")

while True:
    query = input("Enter your question:")
    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke(
        {'context': context, 'question': query}
    )
    llm_response = llm.invoke(final_prompt)

    print(f"\n AI_Answer: {llm_response.content}")