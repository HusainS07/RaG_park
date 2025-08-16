import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_openai import ChatOpenAI

load_dotenv()

# -------------------------
# Load persisted embeddings (do not recompute)
# -------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# -------------------------
# RAG chain setup
# -------------------------
prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-r1:free",
    max_tokens=350,
    temperature=0.7,
    base_url="https://openrouter.ai/api/v1"
)  # type: ignore

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
