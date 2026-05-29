import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from mysite.frontend.kyrgyz_front import check_kyrgyz
from mysite.frontend.speech_front import check_speech
from mysite.frontend.ravdess_front import check_ravdess
from mysite.frontend.gtzan_front import check_gtzan
from mysite.frontend.urban_front import check_urban
from mysite.frontend.esc_50_front import check_esc


with st.sidebar:
    name = st.radio('DL Audios:', ['Info', 'KyrgyzCommands', 'SpeechCommands', 'RAVDESS',
                                   'GTZAN', 'UrbanSound8K', 'ESC-50'])


if name == 'Info':
    st.title('Welcome to Deep Learning Models')
    st.markdown("""
    * **SpeechCommands** — Recognition of short English speech commands (yes, no, stop, go, up, down, etc.)
    * **GTZAN** — Music genre classification (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
    * **UrbanSound8K** — Urban sound classification (car horn, dog bark, siren, street music, drilling, etc.)
    * **KyrgyzSpeechCommands** — Recognition of short Kyrgyz speech commands (own dataset)
    * **RAVDESS** — Speech emotion recognition (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
    """)


elif name == 'KyrgyzCommands':
    check_kyrgyz()

elif name == 'SpeechCommands':
    check_speech()

elif name == 'RAVDESS':
    check_ravdess()

elif name == 'GTZAN':
    check_gtzan()

elif name == 'UrbanSound8K':
    check_urban()

elif name == 'ESC-50':
    check_esc()


