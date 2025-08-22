import whisper
import torch
from TTS.api import TTS

model = whisper.load_model('base')

#translate audio file
# response = model.transcribe('pilar2.mp3', language='es', task='translate')

#transcription audio file - define task 
reponse_transcription = model.transcribe('pilar2.mp3', language='es', task='transcribe')
text_transcription = reponse_transcription['text']

device = "cuda" if torch.cuda.is_available() else "cpu"
print(TTS().list_models())

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

wav = tts.tts(text_transcription, language="en")
tts.tts_to_file(text_transcription, language="en", file_path="output.wav")





