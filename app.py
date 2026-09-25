import os
import subprocess
import tempfile
import streamlit as st
import imageio_ffmpeg
from groq import Groq

# Page Configuration
st.set_page_config(
    page_title="Multilingual Video to Urdu Translator",
    page_icon="🎬",
    layout="wide"
)

# Custom Styling for Text Rendering
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


def extract_audio_ffmpeg(video_path: str, output_audio_path: str) -> bool:
    """
    Extracts complete, unclipped audio from video using FFmpeg directly.
    Converts to 16kHz Mono MP3 which is optimal for Whisper speech recognition.
    """
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_exe,
            "-y",                     # Overwrite output without asking
            "-i", video_path,         # Input video file
            "-vn",                    # Strip video stream
            "-acodec", "libmp3lame",  # Standard MP3 codec
            "-ar", "16000",           # 16kHz sample rate for Whisper
            "-ac", "1",               # Convert to Mono to avoid channel issues
            "-b:a", "64k",            # 64kbps bitrate
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
        st.error(f"FFmpeg audio extraction error: {str(e)}")
        return False


def transcribe_audio_with_groq(client: Groq, audio_path: str, lang_code: str = None) -> str:
    """
    Transcribes audio with Groq Whisper-large-v3.
    Uses temperature=0.0 to prevent hallucinations and premature cutoffs.
    """
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

            # Extract full text across all segments
            if hasattr(transcription, "text"):
                return transcription.text.strip()
            elif isinstance(transcription, dict) and "text" in transcription:
                return transcription["text"].strip()
            return str(transcription).strip()
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""


def translate_text_to_urdu(client: Groq, original_text: str, source_language: str) -> str:
    """
    Translates the full transcript into Urdu using llama-3.1-8b-instant on Groq.
    """
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


# Sidebar: Configuration
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
    st.markdown("""
    **Supported Source Languages:**
    - Arabic (العربية)
    - English
    - Persian (فارسی)
    - Turkish (Türkçe)

    **Target Output:**
    - Urdu Transcript (اردو ترجمہ)
    """)


# Main Interface
st.title("Multilingual Video to Urdu Audio Translator")
st.write(
    "Upload a video file (Arabic, English, Persian, or Turkish). "
    "The audio will be extracted, fully transcribed via Whisper-large-v3, and translated into fluent Urdu."
)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=["mp4", "mkv", "mov", "avi", "webm"],
        help="Upload short or long videos."
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
        st.warning("Please provide a valid Groq API Key to proceed.")
    else:
        if st.button("Process Full Video", type="primary"):
            client = Groq(api_key=groq_api_key)

            with tempfile.TemporaryDirectory() as temp_dir:
                video_suffix = os.path.splitext(uploaded_file.name)[1]
                temp_video_path = os.path.join(temp_dir, f"input_video{video_suffix}")
                temp_audio_path = os.path.join(temp_dir, "extracted_audio.mp3")

                # Save uploaded buffer
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Step 1: Extract Audio via direct FFmpeg
                with st.spinner("Extracting complete audio track with FFmpeg..."):
                    audio_success = extract_audio_ffmpeg(temp_video_path, temp_audio_path)

                if not audio_success or not os.path.exists(temp_audio_path):
                    st.error("Audio extraction failed. Please ensure the video has an active audio stream.")
                else:
                    # Provide audio preview so you can verify the extracted duration
                    with st.expander("Extracted Audio Track (Click to play/verify duration)"):
                        with open(temp_audio_path, "rb") as audio_file:
                            st.audio(audio_file.read(), format="audio/mp3")

                    # Step 2: Transcribe via Groq Whisper
                    with st.spinner("Transcribing full speech with Groq Whisper..."):
                        lang_code = language_mapping[selected_language]
                        transcript = transcribe_audio_with_groq(client, temp_audio_path, lang_code)

                    if not transcript:
                        st.warning("No speech could be recognized in the audio track.")
                    else:
                        # Step 3: Translate to Urdu
                        with st.spinner("Translating transcript into fluent Urdu..."):
                            urdu_translation = translate_text_to_urdu(client, transcript, selected_language)

                        st.success("Complete processing finished successfully!")

                        tab_urdu, tab_original = st.tabs(["Urdu Translation", "Original Transcript"])

                        with tab_urdu:
                            st.subheader("Urdu Text")
                            st.markdown(f'<div class="urdu-text-container">{urdu_translation}</div>', unsafe_allow_html=True)
                            st.download_button(
                                label="Download Urdu Translation",
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
