import whisper
import soundfile as sf
import numpy as np
import librosa 

model = whisper.load_model("base")
audio,sr = sf.read('Recoding.wav')

if len(audio.shape) > 1:
    audio = np.mean(audio,axis=1)

if sr != 16000:
    audio = librosa.resample(audio,orig_sr=sr,target_sr=16000)

audio = audio.astype(np.float32)


print('-'*50)
print('audo loaded successfully')
print('-'*50)

result = model.transcribe(audio)
print(result["text"])
