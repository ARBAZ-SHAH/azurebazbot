from fastapi import FastAPI, Query
import uvicorn
import ollama
import logging
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Chat State Model
class ChatState(BaseModel):
    messages: list[str]

# Function to generate chatbot response
def assistant(state: ChatState):
    try:
        user_message = state.messages[-1]
        logging.info(f"Received user message: {user_message}")

        # Generate response from LLaMA model
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": user_message}]
        )

        # Extract bot response
        bot_response = response["message"]["content"] if "message" in response else "I didn't understand that."
        logging.info(f"Bot Response: {bot_response}")

        return ChatState(messages=[bot_response])

    except Exception as e:
        logging.error(f"Error in assistant: {str(e)}")
        return ChatState(messages=["Error in processing response."])

# Function to summarize response
def summarizer(state: ChatState):
    try:
        if len(state.messages) < 1:
            return state

        last_response = state.messages[-1]
        summary_prompt = f"Summarize this response in exactly 3 concise sentences: {last_response}"
        logging.info(f"Summarization Request: {summary_prompt}")

        response = ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": summary_prompt}]
        )

        # Extract summary
        summary = response["message"]["content"] if "message" in response else None
        logging.info(f"Summary Response: {summary}")

        return ChatState(messages=[state.messages[-1], summary]) if summary else state

    except Exception as e:
        logging.error(f"Error in summarizer: {str(e)}")
        return ChatState(messages=[state.messages[-1], "Summary generation failed."])

# API Home Route
@app.get("/")
def home():
    return {"message": "Chatbot API is running! Use /chat?message=Hello"}

# API Chat Route
@app.get("/chat")
def chat(message: str = Query(..., description="User input message")):
    try:
        logging.info(f"Received API request: {message}")
        
        # Get bot response
        response = assistant(ChatState(messages=[message]))
        
        # Generate summary
        summary_response = summarizer(response)
        bot_response = summary_response.messages[0]
        summary = summary_response.messages[1] if len(summary_response.messages) > 1 else "No summary available."

        # ✅ Ensure correct JSON response (NO `json.dumps()`)
        return {
            "bot": bot_response,
            "summary": summary
        }

    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        return {"error": str(e)}

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
