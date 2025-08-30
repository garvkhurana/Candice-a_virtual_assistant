import os
import sounddevice as sd
from scipy.io import wavfile
from dataclasses import dataclass
from groq import Groq
from dotenv import load_dotenv
from llm_response import get_llm_response  

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@dataclass
class TTSConfig:
    model_name: str = 'playai-tts'
    voice_name: str = 'Gail-PlayAI'
    output_format: str = 'wav'
    speech_file_path: str = 'output.wav'

class TTS:
    def __init__(self):
        self.config = TTSConfig()
        self.model = self.config.model_name
        self.voice_name = self.config.voice_name
        self.output_format = self.config.output_format
        self.path = self.config.speech_file_path

    def generate_tts(self, text: str):
        print("Generating audio from LLM response...")

        response = client.audio.speech.create(
            model=self.model,
            voice=self.voice_name,
            input=text,
            response_format=self.output_format
        )

        response.write_to_file(self.path)
        print(f"Audio saved at: {self.path}")

        self.play_audio(self.path)

    def play_audio(self, path):
        samplerate, data = wavfile.read(path)
        sd.play(data, samplerate)
        sd.wait()