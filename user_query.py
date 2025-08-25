import whisper 
from dataclasses import dataclass
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os

model=None
@dataclass
class whisper_config:
    whisper_model_name:str='tiny'
    duration:int=8
    fs:int=16000
    channels:int=1
    dtype:str='int16'



class speech_to_text:
    def record_and_transcribe(self,config=whisper_config()):
        self.model=config.whisper_model_name
        self.duration=config.duration
        self.fs=config.fs
        self.channels=config.channels
        self.dtype=config.dtype


        global model
        if model is None:
            model=whisper.load_model(self.model)
            print(f" Recording for {self.duration} seconds...")
        audio = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=self.channels, dtype=self.dtype)
        sd.wait()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            write(tmp_wav.name, self.fs, audio)
            temp_path = tmp_wav.name

            print(" Transcribing...")
        result = model.transcribe(temp_path, language="en")
        text = result.get("text", "").strip()

        os.remove(temp_path)

        return text
    

if __name__=='__main__':
    whsiper1=speech_to_text()
    print(whsiper1.record_and_transcribe())


        

