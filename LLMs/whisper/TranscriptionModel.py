import whisper

model = whisper.load_model('base')
response = model.transcribe('pilar2.mp3', language='es', task='transcribe')

print(response['text'])

#The use model whisper






