from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import ollama
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import List

########################
# 1) Define a Pydantic model for the conversation state
########################
class ChatState(BaseModel):
    history: List[str] = []

########################
# 2) Minimal usage of StateGraph with a typed schema
########################
class SimpleChatFlow(StateGraph):
    def __init__(self):
        # Provide the *type* ChatState as state_schema
        super().__init__(state_schema=ChatState)

    def chat(self, user_message: str) -> (str, str):
        """
        Returns a tuple: (full bot answer, 3-line summary).
        """
        # === 1) Main Bot Answer ===
        try:
            response = ollama.chat(
                model="llama3.2:1b",
                messages=[{"role": "user", "content": user_message}]
            )
            bot_content = response.get("message", {}).get("content", "").strip()
            if not bot_content:
                bot_content = "I didn't understand that."
        except Exception as e:
            bot_content = f"Error in Ollama call: {e}"

        # === 2) Summarize the Bot's Answer in exactly 3 lines ===
        summary = "No summary available."
        try:
            summary_prompt = (
                f"Summarize the following text in exactly 3 lines:\n{bot_content}"
            )
            summary_response = ollama.chat(
                model="llama3.2:1b",
                messages=[{"role": "user", "content": summary_prompt}]
            )
            summary_text = summary_response.get("message", {}).get("content", "").strip()
            if summary_text:
                summary = summary_text
        except Exception as e:
            summary = f"Error generating summary: {e}"

        return bot_content, summary

# Create an instance of the flow
chat_flow = SimpleChatFlow()

########################
# 3) Setup FastAPI
########################
app = FastAPI()

# Allow calls from anywhere (for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################
# 4) Request model for /chat
########################
class ChatRequest(BaseModel):
    message: str

########################
# 5) POST route for chatting
########################
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Just call chat_flow.chat() which returns (fullReply, summaryReply)
    full_answer, short_summary = chat_flow.chat(user_input)
    return {
        "response": full_answer,
        "summary": short_summary
    }


########################
# 6) If run as main, uvicorn
########################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
