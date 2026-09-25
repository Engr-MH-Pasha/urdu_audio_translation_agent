import os
import tempfile
import streamlit as st
from moviepy.editor import VideoFileClip
from groq import Groq

# Page Configuration
st.set_page_config(
    page_title="Multilingual Video to Urdu Translator",
    page_icon="🎬",
    layout="wide"
)

# Custom Styling for Clean Display and Urdu RTL Text
st.markdown("""
    <style>
        .urdu-text-container {
            direction: rtl;
            text-align: right;
            font-size: 1.25rem;
            line-height: 2.2;
            padding: 1.5rem;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-right: 5px solid #28a745;
            color: #111;
        }
        .original-text-container {
            direction: ltr;
            text-align: left;
            font-size: 1rem;
            line-height: 1.8;
            padding: 1.5rem;
            background-color: #f1f3f5;
            border-radius: 8px;
            border-left: 5px solid #007bff;
            color: #111;
        }
    </style>
""", unsafe_allow_html=True)


def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    """Extracts audio from an uploaded video file and saves it as MP3."""
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            return False
        video.audio.write_audiofile(
            output_audio_path,
            codec="mp3",
            bitrate="128k",
            logger=None
        )
        video.close()
        return True
    except Exception as e:
        st.error(f"Error during audio extraction: {str(e)}")
        return False


def transcribe_audio_with_groq(client: Groq, audio_path: str, lang_code: str = None) -> str:
    """Transcribes audio using Groq Whisper-large-v3 model."""
    try:
        with open(audio_path, "rb") as file:
            transcription_kwargs = {
                "file": (os.path.basename(audio_path), file.read()),
                "model": "whisper-large-v3",
                "response_format": "text"
            }
            if lang_code:
                transcription_kwargs["language"] = lang_code

            transcription = client.audio.transcriptions.create(**transcription_kwargs)
            return transcription
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return ""


def translate_text_to_urdu(client: Groq, original_text: str, source_language: str) -> str:
    """Translates the transcribed text into fluent, natural Urdu using Llama on Groq."""
    try:
        system_prompt = (
            "You are an expert professional translator specializing in Arabic, English, Persian, and Turkish to Urdu translations.\n"
            "Translate the following speech transcript accurately and idiomatically into fluent, grammatically correct Urdu.\n"
            "Preserve original meaning, nuance, and tone. Output ONLY the Urdu translation without explanations, preambles, or markdown quotes."
        )

        # Using llama-3.1-8b-instant for fast, reliable translation on Groq
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Source Language: {source_language}\n\nTranscript to translate:\n{original_text}"}
            ],
            temperature=0.3,
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error during translation: {str(e)}")
        return ""

# Sidebar: API Key Configuration
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key or set it in Streamlit secrets as GROQ_API_KEY."
    )
    
    # Check for secret if input is empty
    if not groq_api_key and "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]

    st.markdown("---")
    st.markdown("""
    **Supported Source Languages:**
    - Arabic (العربية)
    - English
    - Persian (فارسی)
    - Turkish (Türkçe)
    
    **Target Output:**
    - Fluent Urdu (اردو متن)
    """)


# Main Application Interface
st.title("Multilingual Video to Urdu Audio Translator")
st.write(
    "Upload a video file containing Arabic, English, Persian, or Turkish speech. "
    "The app extracts the audio, transcribes the speech, and delivers a fluent Urdu translation."
)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=["mp4", "mkv", "mov", "avi"],
        help="Upload files up to 200MB."
    )

with col2:
    language_mapping = {
        "Auto-Detect": None,
        "Arabic": "ar",
        "English": "en",
        "Persian (Farsi)": "fa",
        "Turkish": "tr"
    }
    selected_language = st.selectbox("Source Language in Video", list(language_mapping.keys()))

if uploaded_file is not None:
    if not groq_api_key:
        st.warning("Please provide a valid Groq API Key in the sidebar to proceed.")
    else:
        if st.button("Process Video and Translate to Urdu", type="primary"):
            client = Groq(api_key=groq_api_key)

            with tempfile.TemporaryDirectory() as temp_dir:
                video_suffix = os.path.splitext(uploaded_file.name)[1]
                temp_video_path = os.path.join(temp_dir, f"input_video{video_suffix}")
                temp_audio_path = os.path.join(temp_dir, "extracted_audio.mp3")

                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Step 1: Audio Extraction
                with st.spinner("Extracting audio track from video..."):
                    audio_success = extract_audio_from_video(temp_video_path, temp_audio_path)

                if not audio_success:
                    st.error("No valid audio track detected in the uploaded video.")
                else:
                    # Step 2: Speech-to-Text via Groq Whisper
                    with st.spinner("Transcribing speech with Groq Whisper..."):
                        lang_code = language_mapping[selected_language]
                        transcript = transcribe_audio_with_groq(client, temp_audio_path, lang_code)

                    if transcript:
                        # Step 3: Translation into Urdu via Groq LLM
                        with st.spinner("Translating transcript into fluent Urdu..."):
                            urdu_translation = translate_text_to_urdu(client, transcript, selected_language)

                        st.success("Processing completed successfully!")

                        tab_urdu, tab_original = st.tabs(["Urdu Translation", "Original Transcript"])

                        with tab_urdu:
                            st.subheader("Urdu Text")
                            st.markdown(f'<div class="urdu-text-container">{urdu_translation}</div>', unsafe_allow_html=True)
                            st.download_button(
                                label="Download Urdu Text",
                                data=urdu_translation,
                                file_name="urdu_translation.txt",
                                mime="text/plain; charset=utf-8"
                            )

                        with tab_original:
                            st.subheader("Original Speech Transcript")
                            st.markdown(f'<div class="original-text-container">{transcript}</div>', unsafe_allow_html=True)
                            st.download_button(
                                label="Download Original Transcript",
                                data=transcript,
                                file_name="original_transcript.txt",
                                mime="text/plain; charset=utf-8"
                            )
