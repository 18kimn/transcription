# Template repository for transcribing audio locally

Made in advance of a discussion with friends. Feel free to repurpose.

1. Get a [huggingface token](https://huggingface.co/settings/tokens) and put it in a .env file as HF_TOKEN
2. Accept the following agreements / input contact info to access gated models
   a. [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   b. [speaker-diarization-3.1](https://huggingface.co/pyannote/segmentation-3.0)
3. Make a folder "mp4s" and put your MP4 Files there
   a. You can probably adjust make_transcripts.py to work with alternate file types, I just had mp4s
4. Run `poetry install`
   a. You may need to [install poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) if you don't have it
   b. You can also just install pandas, torchaudio, faster_whisper, pyannote.audio, and pydub however you're comfortable with
5. Run `python make_transcripts.py` in the apppropriate virtualenv to make the transcripts
   a. Speaker diarization takes a long time, like 20mins for
   a 1-hr recording. It's faster if you have better hardware or a GPU, but you may need to adjust how the model runs in make_transcripts to account for your hardware.
   b. The Whisper model used is an implementation of distil-large-v3 which is supposed to be pretty good and also pretty fast, but it can still be a bit slow. Depending on your needs you can adjust to "base" or "base.en" or any other model.
6. Once raw transcripts are made, run `poetry run clean_transcripts.py` to clean them up -- basically combine word-level speaker associations into contiguous paragraphs and such.
