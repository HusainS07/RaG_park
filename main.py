from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import rag_chain

app = FastAPI(title="Memory-Efficient RAG Backend")

class Query(BaseModel):
    name: str
    email: str
    query: str

@app.get("/")
def health():
    return {"status": "ok", "message": "RAG backend is running!"}

@app.post("/ask")
async def ask_question(data: Query):
    try:
        response = await rag_chain.ainvoke(data.query)
        if not response or len(response.strip()) < 20:
            return {
                "answer": "Sorry, we couldn't find a detailed answer. Try rephrasing or contact support.",
                "matched": False
            }
        return {
            "answer": response.strip(),
            "matched": True
        }
    except Exception as e:
        print("Error:", e)
        return {
            "error": "Something went wrong. Try again later.",
            "matched": False
        }
