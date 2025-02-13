from fastapi import FastAPI, Query
import uvicorn
import ollama
import logging
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph

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

# Initialize LangGraph Memory
memory = ConversationBufferMemory()

# 🔹 Multi-Agent Functions

# ✅ Chat Agent
def chat_agent(state: ChatState):
    """Handles user messages and generates chatbot response."""
    try:
        user_message = state.messages[-1]
        logging.info(f"User: {user_message}")

        # Store user message in memory
        memory.save_context({"input": user_message}, {})

        # Generate response from LLaMA model
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": user_message}]
        )

        bot_response = response["message"]["content"] if "message" in response else "I didn't understand that."
        logging.info(f"Bot: {bot_response}")

        # Store bot response in memory
        memory.save_context({}, {"output": bot_response})

        return ChatState(messages=[bot_response])

    except Exception as e:
        logging.error(f"Error in chat_agent: {str(e)}")
        return ChatState(messages=["Error generating response."])

# ✅ Summarizer Agent
def summarizer_agent(state: ChatState):
    """Summarizes chatbot response into 3 sentences."""
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

        summary = response["message"]["content"] if "message" in response else None
        logging.info(f"Summary: {summary}")

        return ChatState(messages=[state.messages[-1], summary]) if summary else state

    except Exception as e:
        logging.error(f"Error in summarizer_agent: {str(e)}")
        return ChatState(messages=[state.messages[-1], "Summary generation failed."])

# ✅ Memory Agent (Retrieves Chat History)
def memory_agent(state: ChatState):
    """Retrieves previous chat messages for context-aware responses."""
    try:
        history = memory.load_memory_variables({})
        logging.info(f"Chat History: {history}")

        if history.get("output"):
            state.messages.insert(0, history["output"])  # Add last response as context

        return state

    except Exception as e:
        logging.error(f"Error in memory_agent: {str(e)}")
        return state

# 🔹 Define LangGraph Multi-Agent Chat Flow
graph = StateGraph(ChatState)
graph.add_node("memory_agent", memory_agent)
graph.add_node("chat_agent", chat_agent)
graph.add_node("summarizer_agent", summarizer_agent)

# 🔗 Set execution order
graph.set_entry_point("memory_agent")
graph.add_edge("memory_agent", "chat_agent")
graph.add_edge("chat_agent", "summarizer_agent")

# ✅ Compile Multi-Agent Chatbot
chat_chain = graph.compile()

# 🌐 API Routes

@app.get("/")
def home():
    return {"message": "Multi-Agent Chatbot API is running!"}

@app.get("/chat")
def chat(message: str = Query(..., description="User input message")):
    """Handles chat requests and returns bot response + summary."""
    try:
        logging.info(f"Received API request: {message}")

        # Process through LangGraph pipeline
        response = chat_chain.invoke(ChatState(messages=[message]))

        bot_response = response.messages[0]
        summary = response.messages[1] if len(response.messages) > 1 else "No summary available."

        return {
            "bot": bot_response,
            "summary": summary
        }

    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        return {"error": str(e)}

# Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
