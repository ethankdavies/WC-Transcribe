import os
import yt_dlp as youtube_dl
import whisper
import streamlit as st
import shutil
import tempfile
from googleapiclient.discovery import build
from fuzzywuzzy import fuzz
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load Whisper large-v2 model for transcription
whisper_model = whisper.load_model("large-v2")

# Load the T5 model for summarization
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Set up YouTube API key
YOUTUBE_API_KEY = 'AIzaSyCQOiMG3INHF_xzlU2YZGG4gJOTqYp1Uc0'
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Phrase hints for common names and terms
PHRASE_HINTS = {
    "Ruben Gallego": ["Ruben Gallego"],
    "Kari Lake": ["Kari Lake"],
    "Josh Stein": ["Josh Stein"],
    "Mark Robinson": ["Mark Robinson"],
    "Joyce Craig": ["Joyce Craig"],
    "Kelly Ayotte": ["Kelly Ayotte"],
    "Jacky Rosen": ["Jacky Rosen"],
    "Sam Brown": ["Sam Brown"],
    "Sherrod Brown": ["Sherrod Brown"],
    "Bernie Moreno": ["Bernie Moreno"],
    "Bob Casey": ["Bob Casey"],
    "Dave McCormick": ["Dave McCormick"],
    "Kamala Harris": ["Kamala Harris"],
    "Donald Trump": ["Donald Trump"],
    "Tammy Baldwin": ["Tammy Baldwin"],
    "Eric Hovde": ["Eric Hovde"],
    "Mark Halperin": ["Mark Halperin"],
    "Inflation": ["Inflation"],
    "Immigration": ["Immigration"],
    "Greedflation": ["Greedflation"],
    "Economy": ["Economy"],
    "Democrat": ["Democrat"],
    "Republican": ["Republican"],
    "Abortion": ["Abortion"],
    "Reproductive": ["Reproductive"]
}

# Hard-coded common transcription errors and corrections
COMMON_CORRECTIONS = {
    "Kerry Lake": "Kari Lake",
    "Donald Drumpf": "Donald Trump"
}

# Context-based corrections
CONTEXT_CORRECTIONS = {
    "Kerry Lake": ("Kari Lake", ["Republican", "Arizona"]),
}

# Function to replace common words/phrases using fuzzy matching
def replace_common_phrases(transcription, threshold=85):
    for key_phrase, alternatives in PHRASE_HINTS.items():
        for phrase in alternatives:
            words = transcription.split()
            for i, word in enumerate(words):
                if fuzz.ratio(word.lower(), phrase.lower()) >= threshold:
                    words[i] = key_phrase
            transcription = " ".join(words)
    return transcription

# Function to apply hard-coded corrections
def apply_common_corrections(transcription):
    for incorrect, correct in COMMON_CORRECTIONS.items():
        transcription = transcription.replace(incorrect, correct)
    return transcription

# Function to apply context-based corrections
def apply_context_corrections(transcription):
    for incorrect, (correct, context_terms) in CONTEXT_CORRECTIONS.items():
        if incorrect in transcription:
            for term in context_terms:
                if term in transcription:
                    transcription = transcription.replace(incorrect, correct)
                    break
    return transcription

# Function to download and transcribe YouTube videos
def download_and_transcribe(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': '%(id)s.%(ext)s',
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            audio_file = f"{info_dict['id']}.mp3"
            
            # Transcribe the downloaded audio
            result = whisper_model.transcribe(audio_file)
            transcription = result['text']

            # Apply corrections to improve transcription accuracy
            transcription = replace_common_phrases(transcription)
            transcription = apply_common_corrections(transcription)
            transcription = apply_context_corrections(transcription)
            
            os.remove(audio_file)  # Clean up the audio file after transcribing
            return transcription

    except Exception as e:
        return f"Error: {str(e)}"

# Function to summarize the transcription using T5
def summarize_transcription(text):
    # Preprocess the text for summarization
    inputs = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the summary
    summary_ids = t5_model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

# Function to get recent videos for a channel
def get_recent_videos(channel_id):
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=10,
        order="date",
        type="video"
    )
    response = request.execute()
    videos = response['items']
    
    video_list = []
    for video in videos:
        video_id = video['id']['videoId']
        video_title = video['snippet']['title']
        video_list.append((video_title, video_id))
    
    return video_list

# Streamlit Web Interface
st.title("YouTube Channel Transcriber")

# Channel mapping
channel_mapping = {
    "Ruben Gallego": "UCxggVFesZy65a0WBT3_roXQ",
    "Josh Stein": "UCz1XsZYTzudZtHAIQvuEQAQ",
    "Joyce Craig": "UCBt2qbHd5ns7ryv3n0Y_bHw",
    "Jacky Rosen": "UCq2JO4WbdKPvTfcfmHPmWMw",
    "Sherrod Brown": "UCt_l7Nge_872rTm5Jvbo6Mw",
    "Bob Casey": "UCOak7SAWIvog_DN6dRMO3CA",
    "Kamala Harris": "UC0XBsJpPhOLg0k4x9ZwrWzw",
    "Tammy Baldwin": "UC_XjYCRbbI2_TDDjJidwk0Q",
    "2WAY with Mark Halperin": "UCq7OKQb6_1tbA73oSloIiZQ"
}

# Allow user to select a channel
selected_channel = st.selectbox("Select a Channel", list(channel_mapping.keys()))

# Fetch videos if a channel is selected
if selected_channel:
    channel_id = channel_mapping[selected_channel]
    videos = get_recent_videos(channel_id)
    
    # Allow user to select a video from the channel
    video_options = [f"{title} (ID: {video_id})" for title, video_id in videos]
    selected_video = st.selectbox("Select a Video", video_options)
    
    if selected_video:
        video_id = selected_video.split("(ID: ")[1].split(")")[0]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Checkbox to summarize only for 2WAY with Mark Halperin
        summarize_option = False
        if selected_channel == "2WAY with Mark Halperin":
            summarize_option = st.checkbox("Summarize the transcription")

        # Button to trigger transcription
        if st.button("Transcribe"):
            st.write("Transcribing...")
            transcription = download_and_transcribe(video_url)
            
            # If summarization is requested, summarize the transcription
            if summarize_option:
                transcription = summarize_transcription(transcription)
            
            # Display the transcription (or summary)
            st.write("Transcription Result:")
            st.write(transcription)
