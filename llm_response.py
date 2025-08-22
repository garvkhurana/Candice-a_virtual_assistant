from langchain_groq import Chatgroq
import langchain
from langchain_core.prompts.chat import ChatMessagePromptTemplate
from dataclasses import dataclass
import os

@dataclass
class llm_response_config:
    model_name: str= "llama-3.3-70b-versatile"
    message: str ="your name is friday my personal assistant before any response call me with my name that is by MR.GARV or MR.KHURANA , you will give me assistance to my {query} related to the context of the {detected_object} showed by the me on the screen, what your main aim is to provide all the necessary answers asked by me and reply to them in a polite way as if you are my assistant."
    temperature : float = 0.5
    max_completion_tokens : int= 1024,
    top_p: int = 1
    GROQ_API_KEY : str = os.getenv('GROQ_API_KEY')


class llm_response:
    def __init__(self,config=llm_response_config()):
        self.model_name=config.model_name
        self.message=config.message
        self.temprature=config.temperature
        self.max_completion_tokens=config.max_completion_tokens
        self.top_p=config.top_p
        self.GROQ_API_KEY=config.GROQ_API_KEY

        model=Chatgroq(model=self.model_name,max_tokens=self.max_completion_tokens,temprature=self.temprature,api_key=self.GROQ_API_KEY)
        prompt=ChatMessagePromptTemplate(self.message)


        chain=prompt | model


        









