# flake8: noqa
import streamlit as st

from inference_pipeline import (
    AUDIO_OUTPUT_PATH, DEVICE, HIFI_PARAMS, SPEAKERS_TO_IDS,
    TACOTRON_MODEL_PATH, inference_text_to_speech, CleanedTextIsEmptyStringError,
)
from src.web.streamlit_utils import (
    hide_hamburger_menu, st_empty_block, st_header_centered,
)

LANGUAGES = ["English (en-EN)", "Русский (ru-RU)"]
EMOTIONS = ["😄", "😃", "🙂", "😐", "😑", "😒", "😡"]
RUSSIAN_VOICES = ["Игорь", "Ержан"]
ENGLISH_VOICES = sorted(SPEAKERS_TO_IDS.keys())
DEFAULT_USER_INPUT = "50 points to mr Dumbledore"


def layout_app() -> None:

    # header_text = "🥀 EMOtts dEMO 💔"
    header_text = "EmoTTS Project"
    st_header_centered(header_text)
    st_empty_block(2)

    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox(label="👅 Language", options=LANGUAGES)
    with col2:
        voice = st.selectbox(
            label="🗣️ Speaker",
            options=RUSSIAN_VOICES if language == LANGUAGES[1] else ENGLISH_VOICES,
        )
    st_empty_block(2)

    with st.form(key="input_form"):
        emotion = st.select_slider(label="🎨 Emotion", options=EMOTIONS)
        st_empty_block(2)
        input_text = st.text_area(label="📜 What should I say?", value=DEFAULT_USER_INPUT, max_chars=140)
        st_empty_block(2)
        form_submit = st.form_submit_button("Synthesize speech")

    if form_submit:
        with st.spinner("👷‍♂️ Some loading..."):
            # Handle incorrect input
            try:
                # Run inference pipeline
                inference_text_to_speech(
                    input_text=input_text,
                    speaker_id=SPEAKERS_TO_IDS[voice],
                    audio_output_path=AUDIO_OUTPUT_PATH,
                    tacotron_model_path=TACOTRON_MODEL_PATH,
                    hifi_config=HIFI_PARAMS,
                    device=DEVICE
                )
                st.audio(AUDIO_OUTPUT_PATH)
            except CleanedTextIsEmptyStringError:
                st.warning("😔 Looks like input text can not be pronounced")
                st.stop()
            except KeyError:
                st.warning


def main() -> None:
    hide_hamburger_menu()
    layout_app()


if __name__ == "__main__":
    st.set_page_config(
        page_title="EmoTTS Project",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="auto",
    )
    main()
