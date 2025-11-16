# top of app.py (replace existing model load)
import os
import pickle
import requests
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.pkl"

def _download_file(url, dest_path):
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Failed to download model from URL: {e}")
        return False

@st.cache_resource
def load_model():
    # 1) local file
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Failed to load local model.pkl: {e}")
            return None

    # 2) try download from env var (set on Streamlit Cloud as a Secret)
    model_url = os.getenv("MODEL_PKL_URL", None)
    if model_url:
        ok = _download_file(model_url, MODEL_PATH)
        if ok:
            try:
                with open(MODEL_PATH, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                st.error(f"Downloaded model but failed to unpickle: {e}")
                return None

    # 3) if none available, return None and app should show an explanatory message
    return None

# actual model object used later by app
model = load_model()
if model is None:
    st.warning(
        "Model not available. Add a model.pkl to the repo root or set MODEL_PKL_URL "
        "in app secrets/environment pointing to a raw model file URL."
    )
