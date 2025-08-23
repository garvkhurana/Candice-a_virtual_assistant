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
    message: str = (
        "Your name is Friday, my personal assistant. "
        "Before any response, always address me as MR.GARV or MR.KHURANA. "
        "You are assisting me based on the context of the '{detected_object}' shown on the screen "
        "and my current query '{query}'.\n\n"

        "Additionally, here is the memory of my previous interaction:\n"
        "- Previous Query: {prev_query}\n"
        "- Previous Detected Object: {prev_detected_object}\n"
        "- Previous Response: {prev_response}\n\n"

        "Use all of this information to provide the most accurate, relevant, and polite response possible. "
        "Your main aim is to act like my personal assistant and always maintain continuity."
    )
    temperature: float = 0.5
    max_completion_tokens: int = 1024
    top_p: int = 1
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")


class LLMResponse:
    def __init__(self, query: str, detected_object: str, config: LLMResponseConfig = LLMResponseConfig()):
        self.config = config
        self.memory = deque(maxlen=5)

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
            except Exception:
                self.memory = deque(maxlen=5)

    def _save_memory(self):
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(list(self.memory), f, indent=4)
        except Exception as e:
            print(f" Error saving memory: {e}")

    def get_response(self, query: str, detected_object: str):
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

        self._save_memory()

        return response.content

    def get_memory(self):
        return list(self.memory)


if __name__ == "__main__":
    jarvis = LLMResponse(query="", detected_object="")  

    print("\n Friday is ready! Type 'exit' to end the conversation.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "stop"]:
            print("\n Friday: Goodbye MR.GARV! See you soon.")
            break

        detected_object = input("Enter detected object: ")

        answer = jarvis.get_response(query=query, detected_object=detected_object)
        print(f"\nFriday: {answer}\n")
