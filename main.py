from fastapi import FastAPI
import uvicorn
from mysite.api import speech, gtzan, urban, esc_50, ravdess, kyrgyz

audio_app = FastAPI()
audio_app.include_router(kyrgyz.kyrgyz_router)
audio_app.include_router(speech.speech_router)
audio_app.include_router(ravdess.ravdess_router)
audio_app.include_router(gtzan.gtzan_router)
audio_app.include_router(urban.urban_router)
audio_app.include_router(esc_50.esc_router)


if __name__ == "__main__":
    uvicorn.run(audio_app, host="127.0.0.1", port=8000)
