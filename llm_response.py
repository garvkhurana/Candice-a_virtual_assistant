from langchain_groq import ChatGroq
from langchain_core.prompts.chat import ChatPromptTemplate
from dataclasses import dataclass
from collections import deque
from dotenv import load_dotenv
import json
import os

load_dotenv()

MEMORY_FILE = "memory.json"

@dataclass
class LLMResponseConfig:
    model_name: str = "llama-3.3-70b-versatile"
    message: str = ("""
        Your name is Candice, my personal AI assistant.  
Before responding, you must always address me respectfully as either Mr. Garv or Mr. Khurana.  

You are assisting me based on:  
- The **current detected object**: '{detected_object}'  
- My **current query**: '{query}'  

Additionally, here is the **memory** from our previous interaction:  
- **Previous Query:** {prev_query}  
- **Previous Detected Object:** {prev_detected_object}  
- **Previous Response:** {prev_response}  

Using this information, provide a **highly accurate, context-aware, and polite response**.  
Always ensure **continuity** between the past and current conversation, connect relevant details when possible, and keep responses **clear, concise, and assistant-like**.  
Your primary role is to act as my **personal assistant**, maintaining a professional yet natural tone at all times.
    """)
    temperature: float = 0.5
    max_completion_tokens: int = 1024
    top_p: int = 1
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")


class LLMResponse:
    def __init__(self, query: str, detected_object: str, config: LLMResponseConfig = LLMResponseConfig()):
        self.config = config
        self.memory = deque(maxlen=5)
        self.last_detected_object = detected_object

        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing! Set it in your environment variables.")

        self.model = ChatGroq(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_completion_tokens,
            api_key=config.GROQ_API_KEY
        )

        self.prompt = ChatPromptTemplate.from_template(config.message)

        self._load_memory()

    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    data = json.load(f)
                    self.memory = deque(data, maxlen=5)
                    if self.memory:
                        self.last_detected_object = self.memory[-1]["detected_object"]
            except Exception:
                self.memory = deque(maxlen=5)

    def _save_memory(self):
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(list(self.memory), f, indent=4)
        except Exception as e:
            print(f" Error saving memory: {e}")

    def get_response(self, query: str, detected_object: str = None):
        if not detected_object:
            detected_object = self.last_detected_object

        prev_query = self.memory[-1]["query"] if self.memory else "None"
        prev_detected_object = self.memory[-1]["detected_object"] if self.memory else "None"
        prev_response = self.memory[-1]["response"] if self.memory else "None"

        final_prompt = self.prompt.format_messages(
            query=query,
            detected_object=detected_object,
            prev_query=prev_query,
            prev_detected_object=prev_detected_object,
            prev_response=prev_response
        )

        response = self.model.invoke(final_prompt)

        self.memory.append({
            "query": query,
            "detected_object": detected_object,
            "response": response.content
        })
        self.last_detected_object = detected_object
        self._save_memory()

        return response.content

    def forget_memory(self):
        self.memory.clear()
        self.last_detected_object = ""
        try:
            if os.path.exists(MEMORY_FILE):
                os.remove(MEMORY_FILE)
        except Exception as e:
            print(f"Exception while clearing memory: {e}")


if __name__ == "__main__":
    Candice = LLMResponse(query="", detected_object="")

    print("\nCandice is ready! Type 'exit' to end the conversation.\n")

    while True:
        query = input("You: ").strip().lower()

        
        if query in ["exit", "quit", "stop"]:
            print("\nCandice: Goodbye Mr. Garv! See you soon.")
            break

        
        elif query in ["leave it", "forget it", "just forget about it", "forget about the object"]:
            Candice.forget_memory()
            print("Candice: Okay Mr. Garv, I’ve cleared our previous context.")
            continue

       
        if not Candice.last_detected_object:
            detected_object = input("Enter detected object: ").strip()
        else:
            detected_object = None  

        answer = Candice.get_response(query=query, detected_object=detected_object)
        print(f"\nCandice: {answer}\n")
