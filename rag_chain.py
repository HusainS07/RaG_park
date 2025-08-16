import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI



load_dotenv()

# -------------------------
# Embeddings
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# -------------------------
# Chroma Vectorstore (new way)
# -------------------------
vectorstore = Chroma(
    persist_directory="./chroma_db",  # precomputed embeddings path
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# Prompt template
# -------------------------
prompt_template = """
You are an AI assistant. Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)

# -------------------------
# LLM setup
# -------------------------
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-r1:free",
    max_tokens=350,
    temperature=0.7,
    base_url="https://openrouter.ai/api/v1"
)


# -------------------------
# Format documents
# -------------------------
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# -------------------------
# Build RAG chain
# -------------------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
