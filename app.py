import streamlit as st
import torch
import os
import uuid
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client, Client

# ==========================================
# 1. 기본 웹 설정 및 세션 기억력 초기화
# ==========================================
st.set_page_config(page_title="한국어 AI 클라우드 갤러리", page_icon="🇰🇷", layout="wide")
st.title("🇰🇷 멀티모달 AI 클라우드 갤러리 (한국어 특화)")
st.markdown("번역기를 거치지 않는 '한국어 원어민 AI'로 훨씬 정교하게 사진을 검색해 보세요.")

if "display_count" not in st.session_state:
    st.session_state.display_count = 3
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ==========================================
# 2. 🌟 AI 모델 (한국어 전용으로 교체!) 및 DB 연결
# ==========================================
@st.cache_resource
def load_system():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    sb = create_client(url, key)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 🌟 [핵심] 한국어 데이터로 재학습된 국내 오픈소스 모델! (번역기 불필요)
    model_id = "Bingsu/clip-vit-base-patch32-ko"
    
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    return sb, model, processor, device

with st.spinner('🇰🇷 한국어 원어민 AI 엔진을 깨우는 중입니다...'):
    supabase, model, processor, device = load_system()

# ==========================================
# 3. 화면 탭 구성
# ==========================================
tab_search, tab_upload, tab_manage = st.tabs(["🔍 사진 검색", "☁️ 사진 업로드", "🗑️ 갤러리 관리"])

# ------------------------------------------
# [탭 1] 검색 기능 (번역기 제거!)
# ------------------------------------------
with tab_search:
    st.subheader("머릿속에 있는 사진을 텍스트로 찾아보세요")
    query = st.text_input("검색어 입력 (예: 강아지 사진, 영수증, 바다)", key="search_input")
    
    if query:
        if query != st.session_state.last_query:
            st.session_state.display_count = 3
            st.session_state.last_query = query

        with st.spinner('AI가 한국어 검색어를 직접 이해하고 사진을 찾는 중...'):
            try:
                st.info(f"💡 검색 진행 중: **'{query}'** (🇰🇷 한국어 원어민 AI 가동 중)")
                
                # 🌟 번역 없이 한국어(query)를 그대로 AI에게 직행!
                inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
                
                with torch.no_grad():
                    text_outputs = model.get_text_features(**inputs)
                    
                    if isinstance(text_outputs, torch.Tensor):
                        text_tensor = text_outputs
                    elif hasattr(text_outputs, 'text_embeds'):
                        text_tensor = text_outputs.text_embeds
                    elif hasattr(text_outputs, 'pooler_output'):
                        text_tensor = text_outputs.pooler_output
                    else:
                        text_tensor = text_outputs[0]
                    
                    text_tensor = text_tensor / text_tensor.norm(p=2, dim=-1, keepdim=True)
                    query_vector = text_tensor.flatten().cpu().tolist()[:512]
                
                # 🌟 한국어 모델은 점수 기준이 미세하게 다를 수 있으니, 컷오프를 일단 0.20으로 살짝 내립니다.
                response = supabase.rpc("match_images", {
                    "query_embedding": query_vector,
                    "match_threshold": 0.20,  
                    "match_count": 15 
                }).execute()
                
                results = response.data
                
                if results and len(results) > 0:
                    st.success(f"🎉 총 {len(results)}장의 관련 사진을 찾았습니다!")
                    
                    displayed_results = results[:st.session_state.display_count]
                    
                    cols = st.columns(3)
                    for idx, result in enumerate(displayed_results):
                        with cols[idx % 3]:
                            try:
                                st.image(result['file_path'], use_container_width=True)
                                
                                file_size = result.get('file_size_kb', 0) or 0
                                created_date = result.get('created_at', '알 수 없음')[:10] if result.get('created_at') else '최근'
                                
                                st.markdown(f"**{result['file_name']}**")
                                st.caption(f"🎯 유사도: {result['similarity']:.3f} | 💾 {file_size}KB | 📅 {created_date}")
                                
                                img
