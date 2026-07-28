import os
from datetime import timedelta

import pandas as pd
from webvtt import Caption, WebVTT


def format_timestamp(seconds):
    td = timedelta(seconds=float(seconds))
    
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def make_vtt(filename: str): 
    vtt = WebVTT()
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        caption = Caption(
            format_timestamp(row["start"]),
            format_timestamp(row["end"]),
            [row["speaker"], row["text"]]
        )
        vtt.captions.append(caption)
        
    vtt.save(f"vtts/{os.path.basename(filename).replace('csv', 'vtt')}")

for filename in os.listdir('cleaned_transcripts'):
    if not ".csv" in filename:
        continue
    make_vtt(f"cleaned_transcripts/{filename}")
