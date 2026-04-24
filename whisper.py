import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

'''import os

os.system("ffmpeg -i Recording.m4a Recoding.wav")'''

# 1. Store the file content into a variable
import soundfile as sf

# Audio-va manually load pannunga (16000 sampling rate mukkiyam)
audio_array, sampling_rate = sf.read("Recoding.wav")

print('-'*50)
print('audio loaded successfully')
print('-'*50)





model_id = r".\model"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,dtype = torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

print('-'*50)
print('model loaded successfully')
print('-'*50)

processor = AutoProcessor.from_pretrained(model_id)

print('-'*50)
print('processor loaded successfully')
print('-'*50)

pipe = pipeline(
    "automatic-speech-recognition",
    dtype = torch_dtype, 
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor
)

print('-'*50)
print('pipe line created successfully')
print('-'*50)

print('-'*50)
print('now ready to transcribe')
print('-'*50)


result = pipe("Recoding.wav")

print(result['text'])
