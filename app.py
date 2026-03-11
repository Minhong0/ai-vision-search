import streamlit as st
import torch
import os
import uuid
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client, Client
from deep_translator import GoogleTranslator

# 기본 웹 설정
st.set_page_config(page_title="클라우드 AI 갤러리", page_icon="☁️", layout="wide")
st.title("☁️ 멀티모달 AI 클라우드 갤러리")
st.markdown("한국어 자연어로 클라우드에 저장된 사진을 검색하고 다운로드해 보세요.")

# AI 모델 및 DB 연결
@st.cache_resource
def load_system():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    sb = create_client(url, key)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    return sb, model, processor, device

with st.spinner('AI 엔진과 클라우드를 연결하는 중입니다...'):
    supabase, model, processor, device = load_system()

# 화면 탭 구성
tab_search, tab_upload, tab_manage = st.tabs(["🔍 사진 검색", "☁️ 사진 업로드", "🗑️ 갤러리 관리"])

# [탭 1] 검색 기능

with tab_search:
    st.subheader("머릿속에 있는 사진을 텍스트로 찾아보세요")
    query = st.text_input("검색어 입력 (예: 강아지 사진, 영수증, 바다)", key="search_input")
    
    if query:
        with st.spinner('AI가 검색어를 이해하고 사진을 찾는 중...'):
            try:
                translated_query = GoogleTranslator(source='auto', target='en').translate(query)
                st.info(f"💡 AI 인식 검색어: **'{translated_query}'**")
                
                inputs = processor(text=[translated_query], return_tensors="pt", padding=True).to(device)
                
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
                
                response = supabase.rpc("match_images", {
                    "query_embedding": query_vector,
                    "match_threshold": 0.1,  
                    "match_count": 3
                }).execute()
                
                results = response.data
                
                if results and len(results) > 0:
                    st.success(f"🎉 총 {len(results)}장의 관련 사진을 찾았습니다!")
                    
                    cols = st.columns(3)
                    for idx, result in enumerate(results):
                        with cols[idx % 3]:
                            try:
                                st.image(result['file_path'], use_container_width=True)
                                st.caption(f"이름: {result['file_name']} | 유사도: {result['similarity']:.4f}")
                                
                                img_data = requests.get(result['file_path']).content
                                st.download_button(
                                    label="⬇️ 다운로드",
                                    data=img_data,
                                    file_name=result['file_name'],
                                    mime="image/jpeg",
                                    key=f"dl_search_{idx}_{result['file_name']}"
                                )
                            except Exception as e:
                                st.error(f"이미지 로드 실패")
                else:
                    st.warning("⚠️ 비슷한 사진을 찾지 못했습니다. 검색어를 바꿔보세요!")
            except Exception as e:
                st.error(f"❌ 검색 중 에러 발생: {e}")


# [탭 2] 업로드 기능

with tab_upload:
    st.subheader("새로운 사진들을 클라우드에 한 번에 업로드합니다")
    
    # 🌟 핵심: accept_multiple_files=True 옵션 추가
    uploaded_files = st.file_uploader(
        "이미지 파일 선택 (여러 장 드래그 앤 드롭 가능)", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True
    )
    
    if uploaded_files: # 파일이 1개라도 선택되었다면
        st.write(f"총 **{len(uploaded_files)}**장의 사진이 선택되었습니다.")
        
        # 선택된 사진들을 작게 미리보기 (5칸씩 나눠서 보여주기)
        cols = st.columns(5)
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 5]:
                st.image(file, use_container_width=True)
        
        if st.button("🚀 일괄 클라우드 업로드 및 AI 분석"):
            # 🌟 진행 상태를 보여주는 바(Bar) 생성
            progress_text = "업로드 및 분석을 시작합니다..."
            my_bar = st.progress(0, text=progress_text)
            
            success_count = 0
            
            # 선택된 파일들을 하나씩 꺼내서 반복 작업
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    original_filename = uploaded_file.name
                    ext = os.path.splitext(original_filename)[1]
                    safe_filename = f"{uuid.uuid4().hex}{ext}"
                    
                    file_bytes = uploaded_file.getvalue()
                    
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
                        "embedding": vector_list
                    }
                    supabase.table("image_embeddings").insert(insert_data).execute()
                    
                    success_count += 1
                    
                except Exception as e:
                    st.error(f"❌ '{uploaded_file.name}' 처리 중 에러: {e}")
                
                # 🌟 사진 하나 처리할 때마다 프로그레스 바 게이지 채우기
                progress_percent = int(((idx + 1) / len(uploaded_files)) * 100)
                my_bar.progress(progress_percent, text=f"진행 중... ({idx+1}/{len(uploaded_files)} 장 완료)")
            
            st.success(f"✅ 총 {success_count}장의 사진이 성공적으로 클라우드에 저장되었습니다!")

# [탭 3] 관리 및 삭제 기능
with tab_manage:
    st.subheader("🗑️ 클라우드에 저장된 갤러리 관리")
    
    if st.button("🔄 목록 새로고침"):
        st.rerun()

    try:
        records = supabase.table("image_embeddings").select("file_name", "file_path").execute().data
        
        if not records:
            st.info("현재 클라우드에 저장된 사진이 없습니다.")
        else:
            st.write(f"총 **{len(records)}**장의 사진이 저장되어 있습니다.")
            
            for record in records:
                col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
                
                with col1:
                    st.image(record['file_path'], width=100)
                with col2:
                    st.write(f"**파일명:** {record['file_name']}")
                with col3:
                    try:
                        img_data = requests.get(record['file_path']).content
                        st.download_button(
                            label="⬇️ 다운로드",
                            data=img_data,
                            file_name=record['file_name'],
                            mime="image/jpeg",
                            key=f"dl_manage_{record['file_name']}"
                        )
                    except:
                        st.write("다운로드 불가")
                with col4:
                    if st.button("❌ 삭제", key=f"del_{record['file_name']}"):
                        with st.spinner("삭제 중..."):
                            storage_filename = record['file_path'].split('/')[-1]
                            supabase.storage.from_("images").remove([storage_filename])
                            supabase.table("image_embeddings").delete().eq("file_name", record['file_name']).execute()
                            
                            st.success("삭제되었습니다!")
                            st.rerun()
    except Exception as e:
        st.error(f"목록을 불러오는 중 에러가 발생했습니다: {e}")
