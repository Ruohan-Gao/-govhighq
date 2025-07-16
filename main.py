from fastapi import FastAPI, Request
from pydantic import BaseModel
from Rag_model_TEST import run_llm  # Import your wrapper function
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace with specific origin like ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_input: str
    thread_id: str

class ChatResponse(BaseModel):
    response: str  # ✅ Just one string field, matching your return value


# 5. Expose LLM via /chat POST endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    try:
        result = run_llm(payload.user_input, payload.thread_id)
        return {"response": result}
    except Exception as e:
        return {"response": f"❌ Error: {str(e)}"}