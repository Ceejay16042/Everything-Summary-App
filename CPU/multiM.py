import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = 'cuda:0' if torch.cuda.is_available else 'cpu'

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#whisper large-v3 API
model_id = "openai/whisper-large-v3"

# loading the LLM using the from pretrained module and casting to run on a Gpu runtime execution
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

#audio transcription
def audio_transcription(audio_file):
  pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
  result = pipe(audio_file)
  return result['text']

#saving uploaded audio file locally
def save_audio_file(uploaded_file, save_path):
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.read())

#video transcription
def video_transcription(audio_file):
  pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
  result = pipe(audio_file)
  return result['text']

# saving uploaded video file locally
def save_video_file(uploaded_file, save_path):
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.read())


