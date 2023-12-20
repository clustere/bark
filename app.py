import scipy.io.wavfile as wavfile
import gradio as gr
from transformers import pipeline
import os
import numpy as np

def synthesize_speech(text):
    filename = get_filename()
    synthesizer = pipeline("text-to-speech", model="suno/bark-small")
    speech = synthesizer(text, forward_params={"do_sample": True})
    wavfile.write(filename, rate=speech["sampling_rate"], data=speech["audio"].T)
    
    return filename

def get_filename():
    output_dir = ('./output')
    files = os.listdir(output_dir)
    
    idx = len(files) + 1
    filename = f'synth_{idx}.wav'
    while filename in files:
        idx += 1
        filename = f'synth_{idx}.wav'
    
    return f'{output_dir}/{filename}'

def main():
    
    app = gr.Interface(
        fn=synthesize_speech,
        inputs=gr.TextArea(),
        outputs=["audio"],
        title="Summo: Text synthesizer",
        description="This app synthesizes text into speech."
    )
    app.launch()

if __name__ == "__main__":
    main()
