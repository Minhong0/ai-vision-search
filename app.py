import streamlit as st
import torch
import os
import uuid
import requests
import time
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
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())

# ==========================================
# 2. AI 모델 및 DB 연결
# ==========================================
@st.cache_resource
def load_system():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    sb = create_client(url, key)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 순정 한국어 특화 모델
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
# [탭 1] 검색 기능
# ------------------------------------------
with tab_search:
    st.subheader("머릿속에 있는 사진을 텍스트로 찾아보세요")
    query = st.text_input("검색어 입력 (예: 강아지 사진, 영수증, 바다)", key="search_input")
    
    if query:
        if query != st.session_state.last_query:
            st.session_state.display_count = 3
            st.session_state.last_query = query

        with st.spinner('AI가 사진을 찾는 중...'):
            try:
                st.info(f"💡 검색 진행 중: **'{query}'** (🇰🇷 한국어 원어민 AI 가동 중)")
                
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
                
                # 원상복구된 순수 형태 검색 함수 호출
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
                                
                                img_data = requests.get(result['file_path']).content
                                st.download_button(
                                    label="⬇️ 다운로드",
                                    data=img_data,
                                    file_name=result['file_name'],
                                    mime="image/jpeg",
                                    key=f"dl_search_{result['id']}" 
                                )
                            except Exception as e:
                                st.error(f"이미지 로드 실패")
                    
                    if st.session_state.display_count < len(results):
                        st.markdown("---")
                        if st.button("🔽 더 보기 (Load More)", use_container_width=True):
                            st.session_state.display_count += 3
                            st.rerun()
                else:
                    st.warning("⚠️ 비슷한 사진을 찾지 못했습니다. 검색어를 바꿔보세요!")
            except Exception as e:
                st.error(f"❌ 검색 중 에러 발생: {e}")

# ------------------------------------------
# [탭 2] 업로드 기능
# ------------------------------------------
with tab_upload:
    st.subheader("새로운 사진들을 클라우드에 한 번에 업로드합니다")
    
    uploaded_files = st.file_uploader(
        "이미지 파일 선택 (여러 장 드래그 앤 드롭 가능)", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True,
        key=st.session_state.uploader_key
    )
    
    if uploaded_files:
        st.write(f"총 **{len(uploaded_files)}**장의 사진이 선택되었습니다.")
        
        cols = st.columns(5)
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 5]:
                st.image(file, use_container_width=True)
        
        if st.button("🚀 일괄 클라우드 업로드 및 AI 분석"):
            progress_text = "업로드 및 분석을 시작합니다..."
            my_bar = st.progress(0, text=progress_text)
            success_count = 0
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    original_filename = uploaded_file.name
                    ext = os.path.splitext(original_filename)[1]
                    safe_filename = f"{uuid.uuid4().hex}{ext}"
                    
                    file_bytes = uploaded_file.getvalue()
                    file_size_kb = len(file_bytes) // 1024 
                    
                    supabase.storage.from_("images").upload(
                        path=safe_filename, 
                        file=file_bytes, 
                        file_options={"content-type": uploaded_file.type}
                    )
                    
                    public_url = supabase.storage.from_("images").get_public_url(safe_filename)
                    
                    img = Image.open(uploaded_file).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        img_outputs = model.get_image_features(**inputs)
                        
                        if isinstance(img_outputs, torch.Tensor):
                            img_tensor = img_outputs
                        elif hasattr(img_outputs, 'image_embeds'):
                            img_tensor = img_outputs.image_embeds
                        elif hasattr(img_outputs, 'pooler_output'):
                            img_tensor = img_outputs.pooler_output
                        else:
                            img_tensor = img_outputs[0]
                        
                        img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)
                        vector_list = img_tensor.flatten().cpu().tolist()[:512]
                    
                    insert_data = {
                        "file_name": original_filename, 
                        "file_path": public_url,
                        "file_size_kb": file_size_kb, 
                        "embedding": vector_list
                    }
                    supabase.table("image_embeddings").insert(insert_data).execute()
                    
                    success_count += 1
                    
                except Exception as e:
                    st.error(f"❌ '{uploaded_file.name}' 처리 중 에러: {e}")
                
                progress_percent = int(((idx + 1) / len(uploaded_files)) * 100)
                my_bar.progress(progress_percent, text=f"진행 중... ({idx+1}/{len(uploaded_files)} 장 완료)")
            
            st.success(f"✅ 총 {success_count}장의 사진이 성공적으로 저장되었습니다! 화면이 새로고침됩니다.")
            time.sleep(1.5)
            
            st.session_state.uploader_key = str(uuid.uuid4())
            st.rerun()

# ------------------------------------------
# [탭 3] 관리 및 삭제 기능
# ------------------------------------------
with tab_manage:
    st.subheader("🗑️ 클라우드에 저장된 갤러리 관리")
    
    if st.button("🔄 목록 새로고침"):
        st.rerun()

    try:
        records = supabase.table("image_embeddings").select("id", "file_name", "file_path", "file_size_kb", "created_at").execute().data
        
        if not records:
            st.info("현재 클라우드에 저장된 사진이 없습니다.")
        else:
            st.write(f"총 **{len(records)}**장의 사진이 저장되어 있습니다.")
            
            for record in records:
                col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
                
                with col1:
                    st.image(record['file_path'], width=100)
                with col2:
                    file_size = record.get('file_size_kb', 0) or 0
                    created_date = record.get('created_at', '알 수 없음')[:10] if record.get('created_at') else '기존 데이터'
                    
                    st.write(f"**{record['file_name']}**")
                    st.caption(f"💾 {file_size}KB | 📅 {created_date}")
                with col3:
                    try:
                        img_data = requests.get(record['file_path']).content
                        st.download_button(
                            label="⬇️ 다운로드",
                            data=img_data,
                            file_name=record['file_name'],
                            mime="image/jpeg",
                            key=f"dl_manage_{record['id']}" 
                        )
                    except:
                        st.write("다운로드 불가")
                with col4:
                    if st.button("❌ 삭제", key=f"del_{record['id']}"):
                        with st.spinner("삭제 중..."):
                            storage_filename = record['file_path'].split('/')[-1]
                            supabase.storage.from_("images").remove([storage_filename])
                            supabase.table("image_embeddings").delete().eq("id", record['id']).execute()
                            
                            st.success("삭제되었습니다!")
                            st.rerun()
    except Exception as e:
        st.error(f"목록을 불러오는 중 에러가 발생했습니다: {e}")
