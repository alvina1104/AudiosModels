import streamlit as st
import requests

def check_gtzan():

    st.title("GTZAN Music Classifier")

    api_url = "http://127.0.0.1:8000/predict/"

    option = st.radio(
        "Как хотите загрузить аудио?",
        ["Загрузить голосовой",
         "Записать голосовой"])

    audio_file = None

    if option == "Загрузить голосовой":
        audio_file = st.file_uploader("Choose a file",type=["wav", "mp3", "flac"])

    else:
        audio_file = st.audio_input("Запишите голос")

    if audio_file is not None:

        st.audio(audio_file)
        if st.button("Predict"):
            files = {"file": ( audio_file.name,audio_file, "audio/wav")
            }

            try:
                response = requests.post(api_url, files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Model thinks it's: {result['Class']}")

                else:
                    st.error(response.text)

            except Exception as e:
                st.error(f"Connection error: {e}")