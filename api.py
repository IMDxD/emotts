import random

from fastapi import FastAPI
from fastapi.responses import FileResponse

from inference_pipeline import (
    AUDIO_OUTPUT_PATH, DEVICE, HIFI_PARAMS, SPEAKERS_TO_IDS,
    TACOTRON_MODEL_PATH, inference_text_to_speech, CleanedTextIsEmptyStringError,
)


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "EmoTTS Project"}


# To read this from response:
# r = requests.get("http://localhost:8000/audio", stream=True)
# if r.status_code == 200:
#     with open(path, 'wb') as f:
#         for chunk in r:
#             f.write(chunk)
@app.get("/audio", response_class=FileResponse)
async def audio():
    filepath = "data/audio/beep.wav"
    return filepath


@app.get("/tts", response_class=FileResponse)
async def tts(speaker: int, text: str):
    filepath = f"generated-{random.randrange(1_000_000)}.wav"
    inference_text_to_speech(
        input_text=text,
        speaker_id=SPEAKERS_TO_IDS[speaker],
        audio_output_path=AUDIO_OUTPUT_PATH,
        tacotron_model_path=TACOTRON_MODEL_PATH,
        hifi_config=HIFI_PARAMS,
        device=DEVICE,
    )
    return filepath
