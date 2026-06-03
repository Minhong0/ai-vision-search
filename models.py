import torch
import streamlit as st
from transformers import AutoProcessor, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="CLIP 모델 로딩중...")
def load_ai_model():
    processor = AutoProcessor.from_pretrained("Bingsu/clip-vit-large-patch14-ko")
    model = AutoModel.from_pretrained("Bingsu/clip-vit-large-patch14-ko").to(device)
    return processor, model


@st.cache_resource(show_spinner="OCR 모델 로딩중...")
def load_ocr_reader():
    try:
        import easyocr
        return easyocr.Reader(["ko", "en"], gpu=torch.cuda.is_available())
    except ImportError:
        return None


processor, model = load_ai_model()
ocr_reader = load_ocr_reader()
