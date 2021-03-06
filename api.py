from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTasks

from inference_pipeline import DEVICE, inference_text_to_speech
from src.constants import SupportedEmotions, SupportedLanguages

EMOTTS_API_ROUTE = "/tts/emo/v1"
TEST_AUDIO_PATH = "data/testaudio-gs-16b-1c-44100hz.wav"
ARG_TO_LANGUAGE = {
    "ru": SupportedLanguages.russian,
    "en": SupportedLanguages.english,
}
ARG_TO_EMOTION = {
    SupportedEmotions.angry.api_name: SupportedEmotions.angry,
    SupportedEmotions.happy.api_name: SupportedEmotions.happy,
    SupportedEmotions.neutral.api_name: SupportedEmotions.neutral,
    SupportedEmotions.sad.api_name: SupportedEmotions.sad,
    SupportedEmotions.surprise.api_name: SupportedEmotions.surprise,
    SupportedEmotions.very_angry.api_name: SupportedEmotions.very_angry,
    SupportedEmotions.very_happy.api_name: SupportedEmotions.very_happy,
}


app = FastAPI()


def remove_file(path: Path):
    path.unlink(missing_ok=True)


@app.get("/")
async def root():
    return {"message": "EmoTTS Project"}


# To read this from response:
# r = requests.get("http://<host>:<port>/testaudio", stream=True)
# if r.status_code == 200:
#     with open(path, 'wb') as f:
#         for chunk in r:
#             f.write(chunk)
@app.get("/testaudio", response_class=FileResponse)
async def audio():
    filepath = Path(TEST_AUDIO_PATH)
    return filepath


@app.get(EMOTTS_API_ROUTE, response_class=FileResponse)
async def tts(lang: str, emo: str, speaker: str, text: str, bg_tasks: BackgroundTasks):
    language = ARG_TO_LANGUAGE.get(lang)
    emotion = ARG_TO_EMOTION.get(emo)
    if not language or not emotion:
        raise HTTPException(status_code=500, detail=f"OOPS!\nlang:{lang}, emo:{emo}, text:{text}")
    generated_audio_path = Path(f"predictions/generated-{str(uuid4())}.wav")
    inference_text_to_speech(
        language=language,
        input_text=text,
        speaker=speaker,
        emotion=emotion,
        audio_output_path=generated_audio_path,
        device=DEVICE,
    )
    bg_tasks.add_task(remove_file, generated_audio_path)
    return generated_audio_path
