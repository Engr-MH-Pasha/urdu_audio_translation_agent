import os
import subprocess
import tempfile
import streamlit as st
import imageio_ffmpeg
from groq import Groq

# Page Configuration
st.set_page_config(
    page_title="Multilingual Video & Script to Urdu Translator",
    page_icon="🎬",
    layout="wide"
)

# Initialize Session State Variables
if "video_transcript" not in st.session_state:
    st.session_state["video_transcript"] = ""

# Custom CSS for UI and Urdu RTL Text
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
    </style>
""", unsafe_allow_html=True)


def extract_audio_ffmpeg(video_path: str, output_audio_path: str) -> bool:
    """Extracts complete audio track from video using FFmpeg directly."""
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_exe,
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "64k",
            output_audio_path
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        st.error(f"FFmpeg extraction error: {str(e)}")
        return False


def transcribe_audio_with_groq(client: Groq, audio_path: str, lang_code: str = None) -> str:
    """Transcribes audio using Groq Whisper-large-v3 model."""
    try:
        with open(audio_path, "rb") as file:
            transcription_kwargs = {
                "file": (os.path.basename(audio_path), file.read()),
                "model": "whisper-large-v3",
                "response_format": "verbose_json",
                "temperature": 0.0
            }
            if lang_code:
                transcription_kwargs["language"] = lang_code

            transcription = client.audio.transcriptions.create(**transcription_kwargs)

            if hasattr(transcription, "text"):
                return transcription.text.strip()
            elif isinstance(transcription, dict) and "text" in transcription:
                return transcription["text"].strip()
            return str(transcription).strip()
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""


def translate_text_to_urdu(client: Groq, original_text: str, source_language: str) -> str:
    """Translates the input text into fluent, natural Urdu using Llama on Groq."""
    try:
        system_prompt = (
            "You are an expert translator specializing in Arabic, English, Persian, and Turkish to Urdu translations.\n"
            "Translate the entire text accurately into natural, idiomatic Urdu.\n"
            "Do not omit any sentence or thought. Output ONLY the Urdu translation text with no notes, intros, or markdown quotes."
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Source Language: {source_language}\n\nFull Text to Translate:\n{original_text}"}
            ],
            temperature=0.2,
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return ""


# Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key or configure GROQ_API_KEY in Streamlit secrets."
    )

    if not groq_api_key and "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]

    st.markdown("---")
    language_mapping = {
        "Auto-Detect": None,
        "Arabic": "ar",
        "English": "en",
        "Persian (Farsi)": "fa",
        "Turkish": "tr"
    }
    selected_language = st.selectbox("Source Language", list(language_mapping.keys()))

    st.markdown("---")
    st.markdown("""
    **Features:**
    - Transcribe Video Audio (Whisper Large v3)
    - Manual Script Input & Direct Translation
    - Editable Transcript before Translation
    - Accurate Urdu Output (Llama 3.1)
    """)


# Main Interface
st.title("Multilingual Video & Script to Urdu Translator")
st.write("Convert Arabic, English, Persian, or Turkish video speech or written transcripts into fluent Urdu.")

if not groq_api_key:
    st.warning("Please provide a valid Groq API Key in the sidebar to proceed.")
else:
    client = Groq(api_key=groq_api_key)

    tab_video, tab_direct_text = st.tabs(["📹 Video Upload Mode", "✍️ Direct Script / Text Mode"])

    # --- TAB 1: VIDEO UPLOAD MODE ---
    with tab_video:
        uploaded_file = st.file_uploader(
            "Upload Video File",
            type=["mp4", "mkv", "mov", "avi", "webm"],
            key="video_uploader"
        )

        if uploaded_file is not None:
            if st.button("Extract & Transcribe Audio", type="secondary"):
                with tempfile.TemporaryDirectory() as temp_dir:
                    video_suffix = os.path.splitext(uploaded_file.name)[1]
                    temp_video_path = os.path.join(temp_dir, f"input_video{video_suffix}")
                    temp_audio_path = os.path.join(temp_dir, "extracted_audio.mp3")

                    with open(temp_video_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with st.spinner("Extracting complete audio track with FFmpeg..."):
                        audio_success = extract_audio_ffmpeg(temp_video_path, temp_audio_path)

                    if not audio_success:
                        st.error("Audio extraction failed. Please ensure the video has an active audio track.")
                    else:
                        with st.spinner("Transcribing speech with Groq Whisper..."):
                            lang_code = language_mapping[selected_language]
                            transcript_result = transcribe_audio_with_groq(client, temp_audio_path, lang_code)
                            st.session_state["video_transcript"] = transcript_result

        # Editable Transcript Section
        st.markdown("### Original Transcript (Editable)")
        editable_transcript = st.text_area(
            "Review or edit the transcribed text before translation:",
            value=st.session_state["video_transcript"],
            height=180,
            key="video_text_editor"
        )

        if st.button("Translate Transcript to Urdu", type="primary", key="btn_translate_video"):
            if not editable_transcript.strip():
                st.warning("No transcript available. Please extract audio or type the text above.")
            else:
                with st.spinner("Translating text into fluent Urdu..."):
                    urdu_output = translate_text_to_urdu(client, editable_transcript, selected_language)

                if urdu_output:
                    st.markdown("### Urdu Translation")
                    st.markdown(f'<div class="urdu-text-container">{urdu_output}</div>', unsafe_allow_html=True)
                    st.download_button(
                        label="Download Urdu Text",
                        data=urdu_output,
                        file_name="urdu_translation.txt",
                        mime="text/plain; charset=utf-8"
                    )

    # --- TAB 2: DIRECT TEXT / SCRIPT MODE ---
    with tab_direct_text:
        st.write("If you don't have a video, write or paste the script below to translate directly into Urdu:")
        direct_input_text = st.text_area(
            "Enter original script here (Arabic, English, Persian, Turkish):",
            height=220,
            placeholder="Type or paste the speech transcript here..."
        )

        if st.button("Translate Text to Urdu", type="primary", key="btn_translate_direct"):
            if not direct_input_text.strip():
                st.warning("Please enter some text to translate.")
            else:
                with st.spinner("Translating script into fluent Urdu..."):
                    direct_urdu_output = translate_text_to_urdu(client, direct_input_text, selected_language)

                if direct_urdu_output:
                    st.markdown("### Urdu Translation")
                    st.markdown(f'<div class="urdu-text-container">{direct_urdu_output}</div>', unsafe_allow_html=True)
                    st.download_button(
                        label="Download Urdu Text",
                        data=direct_urdu_output,
                        file_name="direct_urdu_translation.txt",
                        mime="text/plain; charset=utf-8"
                    )
