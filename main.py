from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_chain import rag_chain

app = FastAPI()

class Query(BaseModel):
    name: str
    email: str
    query: str

@app.post("/ask")
async def ask_question(data: Query):
    try:
        response = await rag_chain.ainvoke(data.query)
        if not response or len(response.strip()) < 20:
            return {
                "answer": "Sorry, we couldn't find a detailed answer to your query. Please try rephrasing or contact our support team.",
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