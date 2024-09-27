import os

import pandas as pd
import torchaudio
from dotenv import dotenv_values
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pydub import AudioSegment

HF_TOKEN = dotenv_values("../../.env")["HF_TOKEN"]

# Must accept at https://huggingface.co/pyannote/segmentation-3.0 and https://huggingface.co/pyannote/speaker-diarization-3.1
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", 
    use_auth_token=HF_TOKEN
)
whisper_model = WhisperModel("distil-large-v3")

if not os.path.exists('temp_audio'):
    os.makedirs('temp_audio')
if not os.path.exists('transcripts'):
    os.makedirs('transcripts')
if not os.path.exists('cleaned_transcripts'):
    os.makedirs('transcripts')

def transcribe_mp4(filename):
    """
    Combines Pyannote and Whisper to transcribe segments of an interview with speaker annotation
    """
    export_filename = f"transcripts/{os.path.basename(filename).replace('.mp4', '')}.csv"
    if os.path.exists(export_filename):
       return  
    
    # Convert to WAV, since Pyannote doesn't support mp4s
    audio = AudioSegment.from_file(filename, format="mp4")

    print("Running whisper for full transcript...")
    whisper_result, _ = whisper_model.transcribe(filename, word_timestamps=True)
    segments = list(whisper_result)

    print("Preloading audio into memory...")
    audio.set_frame_rate(8000)
    audio.export("temp_audio/full.wav", format="wav")
    waveform, sample_rate = torchaudio.load("temp_audio/full.wav")

    print("Starting diarization...")
    with ProgressHook() as hook:
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
    print("Diarization complete")
 
    speaker_transcriptions = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        for segment in segments:
            for word in segment.words: 
                if word.start < turn.end and word.end > turn.start:
                    speaker_transcriptions.append({
                        'start': word.start,
                        'end': word.end,
                        'speaker': speaker,
                        'text': word.word,
                    })
    pd.DataFrame(speaker_transcriptions).to_csv(export_filename, index=False)
            
    
for filename in os.listdir('mp4s'):
    transcribe_mp4(f"mp4s/{filename}")