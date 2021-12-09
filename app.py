# flake8: noqa
from dataclasses import dataclass
from pathlib import Path
import random
import requests
import uuid

import streamlit as st

from api import EMOTTS_API_ROUTE
from inference_pipeline import inference_text_to_speech, CleanedTextIsEmptyStringError
from src.web.streamlit_utils import hide_hamburger_menu, st_empty_block, st_header_centered
from src.constants import Language, Emotion, SupportedLanguages, SupportedEmotions


SELECTOR_TO_EMO = {
    "😃": SupportedEmotions.very_happy,
    "🙂": SupportedEmotions.happy,
    "😐": SupportedEmotions.sad,
    "😒": SupportedEmotions.angry,
    "😡": SupportedEmotions.very_angry,
}

SELECTOR_TO_LANG = {
    "🇷🇺 Russian (ru-RU)": SupportedLanguages.russian,
    "🇺🇸 English (en-EN)": SupportedLanguages.english,
}

LOADING_PHRASES = [
    "👷‍♂️ Building language corpuses...",
    "😋 Dreaming about free GPU hosting...",
    "💫 Permuting tensors...",
]

DEFAULT_USER_INPUT = "Я усиленно обогреваю серверную в эту холодные зимние дни"


@dataclass
class AppModes:
    api_connector: str = "api"
    standalone: str = "standalone"


APP_MODE = AppModes.api_connector
HOST = "172.28.108.93"
PORT = 8080
EMOTTS_API_ENDPOINT = f"http://{HOST}:{PORT}{EMOTTS_API_ROUTE}"


def run_in_api_connector_mode(language: Language, emotion: Emotion, input_text: str, audio_output_path: Path) -> None:
    query = f"{EMOTTS_API_ENDPOINT}?lang={language.api_name}&emo={emotion.api_name}&text={input_text}"
    r = requests.get(query, stream=True)
    if r.status_code == 200:
        with open(audio_output_path, 'wb') as f:
            for chunk in r:
                f.write(chunk)


def run_in_standalone_mode(language: Language, emotion: Emotion, input_text: str, audio_output_path: Path) -> None:
    inference_text_to_speech(
        language=language,
        input_text=input_text,
        emotion=emotion,
        audio_output_path=audio_output_path,
    )


def layout_app() -> None:

    # header_text = "🥀 EMOtts dEMO 💔"
    header_text = "EmoTTS Project"
    st_header_centered(header_text)
    st_empty_block(2)

    col1, col2 = st.columns(2)
    with col1:
        language_name = st.selectbox(label="👅 Language", options=SELECTOR_TO_LANG.keys())
        language = SELECTOR_TO_LANG.get(language_name)
    with col2:
        emotion_name = st.select_slider(label="🎨 Emotion", options=SELECTOR_TO_EMO.keys())
        emotion = SELECTOR_TO_EMO.get(emotion_name)
    st_empty_block(2)

    with st.form(key="input_form"):
        input_text = st.text_area(label="📜 What should I say?", value=DEFAULT_USER_INPUT, max_chars=140)
        st_empty_block(2)
        form_submit = st.form_submit_button("Synthesize speech")

    if form_submit:
        loading_phrase = random.choice(LOADING_PHRASES)
        with st.spinner(loading_phrase):
            # Handle incorrect input
            audio_output_path = Path(f"generated-{uuid.uuid4()}.wav")
            try:
                # Run inference pipeline
                if APP_MODE == AppModes.standalone:
                    run_in_standalone_mode(
                        language=language,
                        input_text=input_text,
                        emotion=emotion,
                        audio_output_path=audio_output_path,
                    )
                elif APP_MODE == AppModes.api_connector:
                    run_in_api_connector_mode(
                        language=language,
                        input_text=input_text,
                        emotion=emotion,
                        audio_output_path=audio_output_path,
                    )
                st.audio(audio_output_path)
            except CleanedTextIsEmptyStringError:
                st.warning("😔 Looks like input text can not be pronounced")
                st.stop()
            # except Exception:
            #     st.error("Oops! Forget about it and hit F5 🙈")
            #     st.stop()


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
