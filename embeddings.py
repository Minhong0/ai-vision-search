import torch
import numpy as np
import streamlit as st
from PIL import Image
from models import processor, model, device


@st.cache_data(show_spinner=False)
def get_text_embedding(query_text: str) -> list:
    """같은 검색어는 임베딩을 재계산하지 않고 캐시에서 반환"""
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        t = out if isinstance(out, torch.Tensor) else (
            out.text_embeds if hasattr(out, "text_embeds") else (
                out.pooler_output if hasattr(out, "pooler_output") else out[0]
            )
        )
        t = t / t.norm(p=2, dim=-1, keepdim=True)
    return t.flatten().cpu().tolist()[:768]


def get_image_embedding(img: Image.Image) -> list:
    """PIL Image → 정규화된 임베딩 벡터 (Image는 unhashable이므로 캐시 불가)"""
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        t = out if isinstance(out, torch.Tensor) else (
            out.image_embeds if hasattr(out, "image_embeds") else (
                out.pooler_output if hasattr(out, "pooler_output") else out[0]
            )
        )
        t = t / t.norm(p=2, dim=-1, keepdim=True)
    return t.flatten().cpu().tolist()[:768]
