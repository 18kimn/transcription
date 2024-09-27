import os

import pandas as pd


def clean_transcript(filename):
    df = pd.read_csv(filename, index_col=0).drop_duplicates().reset_index().reset_index()
    def create_contiguous_block(group):
        group['contiguous_block'] = (group['index'].diff() != 1).cumsum()
        return group

    df = df.groupby('speaker', group_keys=False).apply(create_contiguous_block)
    result = df.groupby(['speaker', 'contiguous_block']).agg({
        'start': min,
        'end': max,
        'text': ' '.join    
    }).reset_index().drop(columns=['contiguous_block']).sort_values(by=['start'])

    export_filename = f"cleaned_transcripts/{os.path.basename(filename)}"
    result.to_csv(export_filename, index=False)


for filename in os.listdir('transcripts'):
    clean_transcript(f"transcripts/{filename}")
