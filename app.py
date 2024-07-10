import os
import json
import streamlit as st
from openai import OpenAI

with open('api_key.json', 'r') as key_file:
    api_key = json.load(key_file)['key']

# Initialize OpenAI client
client = OpenAI(api_key=api_key)
# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Function to transcribe audio using OpenAI Whisper
def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
    return response

# Function to summarize text using OpenAI GPT-3.5-turbo
def summarize_text(text):
    summary_prompt = (
        "Please summarize the following transcription in a concise manner:\n\n"
        f"{text}"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": summary_prompt
            }
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("Audio File Transcription and Summarization")

# File uploader for audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "flac", "webm"])

if uploaded_file is not None:
    # Save the uploaded file to disk
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(f"File uploaded successfully: {file_path}")

    with st.spinner('Transcribing audio...'):
        transcription_response = transcribe_audio(file_path)
        transcription_text = transcription_response.text

    st.subheader("Full Transcription")
    st.write(transcription_text)

    with st.spinner('Summarizing transcription...'):
        summary_text = summarize_text(transcription_text)

    st.subheader("Summary")
    st.write(summary_text)




































