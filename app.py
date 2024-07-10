import os
import json
import streamlit as st
from openai import OpenAI

# Fetch the API key from environment variables
api_key = st.secrets["API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

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

# Function to analyze the conversation and extract details
def analyze_conversation(transcription_text):
    analysis_prompt = (
        "After organizing the conversation, take note of every detail provided by the customer including their name, contact information, shipping address, and products ordered. "
        "Format this information clearly and concisely. Finally, evaluate the overall tone of the conversation and determine whether it was positive or negative.\n\n"
        "Here is the format to use:\n\n"
        "Conversation:\n"
        "{conversation}\n\n"
        "Details:\n"
        "- Customer Name: [Customer Name]\n"
        "- Contact Information: [Contact Information]\n"
        "- Shipping Address: [Shipping Address]\n"
        "- Products Ordered: [Products Ordered]\n\n"
        "Evaluation:\n"
        "The overall tone of the conversation was [Positive/Negative] because [reason]."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": analysis_prompt
            },
            {
                "role": "user",
                "content": transcription_text
            }
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("Audio File Transcription and Summarization")

# File uploader for audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "flac", "webm"])

if uploaded_file is not None:
    # Ensure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

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

    with st.spinner('Analyzing transcription...'):
        analysis_text = analyze_conversation(transcription_text)

    st.subheader("Detailed Analysis and Evaluation")
    st.write(analysis_text)



