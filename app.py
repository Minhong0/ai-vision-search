import streamlit as st
import torch
import os
import uuid
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client, Client

# ==========================================
# 1. 기본 웹 설정
# ==========================================
st.set_page_config(page_title="클라우드 AI 갤러리", page_icon="☁️", layout="wide")
st.title("☁️ 멀티모달 AI 클라우드 갤러리")
st.markdown("자연어로 클라우드에 저장된 사진을 검색하고 관리해 보세요.")

# ==========================================
# 2. AI 모델 및 DB 연결 (캐싱 적용)
# ==========================================
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

# ==========================================
# 3. 화면 탭 구성
# ==========================================
tab_search, tab_upload, tab_manage = st.tabs(["🔍 사진 검색", "☁️ 사진 업로드", "🗑️ 갤러리 관리"])

# ------------------------------------------
# [탭 1] 검색 기능
# ------------------------------------------
with tab_search:
    st.subheader("머릿속에 있는 사진을 텍스트로 찾아보세요")
    query = st.text_input("검색어 입력 (예: a photo of an ocean, a dog)", key="search_input")
    
    if query:
        with st.spinner('클라우드에서 사진을 찾는 중...'):
            try:
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
                
                response = supabase.rpc("match_images", {
                    "query_embedding": query_vector,
                    "match_threshold": 0.2,  
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
                                # DB에 저장된 예쁜 '원래 이름(한글)'을 출력합니다!
                                st.caption(f"이름: {result['file_name']} | 유사도: {result['similarity']:.4f}")
                            except Exception as e:
                                st.error(f"이미지 로드 실패")
                else:
                    st.warning("⚠️ 비슷한 사진을 찾지 못했습니다. 검색어를 바꿔보세요!")
            except Exception as e:
                st.error(f"❌ 검색 중 에러 발생: {e}")

# ------------------------------------------
# [탭 2] 업로드 기능
# ------------------------------------------
with tab_upload:
    st.subheader("새로운 사진을 클라우드에 업로드합니다")
    uploaded_file = st.file_uploader("이미지 파일 선택", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, width=300)
        
        if st.button("🚀 클라우드 업로드 및 AI 분석"):
            with st.spinner("AI 분석 및 클라우드 저장 중..."):
                try:
                    # 🌟 [한글 파일명 해결 로직]
                    original_filename = uploaded_file.name # 본명 (예: 귀여운 강아지.jpg)
                    ext = os.path.splitext(original_filename)[1]
                    safe_filename = f"{uuid.uuid4().hex}{ext}" # 가명 (예: a1b2c3.jpg)
                    
                    file_bytes = uploaded_file.getvalue()
                    
                    # 1. 클라우드에는 안전한 '가명'으로 업로드!
                    supabase.storage.from_("images").upload(
                        path=safe_filename, 
                        file=file_bytes, 
                        file_options={"content-type": uploaded_file.type}
                    )
                    
                    # 2. Public URL 가져오기
                    public_url = supabase.storage.from_("images").get_public_url(safe_filename)
                    
                    # 3. AI 분석
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
                    
                    # 4. DB 장부에는 우리가 볼 수 있게 '본명(한글)'으로 기록!
                    insert_data = {
                        "file_name": original_filename, 
                        "file_path": public_url, 
                        "embedding": vector_list
                    }
                    supabase.table("image_embeddings").insert(insert_data).execute()
                    
                    st.success("✅ 클라우드 저장 완료! 이제 한글 이름으로도 완벽하게 관리됩니다.")
                except Exception as e:
                    st.error(f"❌ 처리 중 에러 발생: {e}")

# ------------------------------------------
# [탭 3] 관리 및 삭제 기능
# ------------------------------------------
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
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.image(record['file_path'], width=100)
                with col2:
                    st.write(f"**파일명:** {record['file_name']}") # 예쁜 한글 이름 출력
                with col3:
                    if st.button("❌ 삭제", key=f"del_{record['file_name']}"):
                        with st.spinner("삭제 중..."):
                            # 🌟 [삭제 해결 로직] URL에서 '가명(안전한 이름)'을 추출해서 클라우드에서 지움!
                            storage_filename = record['file_path'].split('/')[-1]
                            supabase.storage.from_("images").remove([storage_filename])
                            
                            # DB에서는 '본명(한글 이름)'을 기준으로 장부 기록을 지움!
                            supabase.table("image_embeddings").delete().eq("file_name", record['file_name']).execute()
                            
                            st.success("삭제되었습니다!")
                            st.rerun()
    except Exception as e:
        st.error(f"목록을 불러오는 중 에러가 발생했습니다: {e}")
