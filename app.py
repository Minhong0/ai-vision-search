import streamlit as st
import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from supabase import create_client, Client

# ==========================================
# 1. 기본 웹 설정 및 로딩 화면
# ==========================================
st.set_page_config(page_title="AI 갤러리 검색엔진", page_icon="🔍", layout="wide")
st.title("🔍 멀티모달 AI 사진 검색 엔진")
st.markdown("자연어로 사진을 검색하고, 새로운 사진을 AI 장부에 추가해 보세요.")

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

with st.spinner('AI 엔진을 예열하는 중입니다... (최초 1회만 소요)'):
    supabase, model, processor, device = load_system()

# ==========================================
# 3. 화면 탭 구성
# ==========================================
tab_search, tab_upload = st.tabs(["🔍 자연어 사진 검색", "☁️ 새 사진 학습(업로드)"])

# ------------------------------------------
# [탭 1] 검색 기능
# ------------------------------------------
with tab_search:
    st.subheader("머릿속에 있는 사진을 텍스트로 찾아보세요")
    query = st.text_input("검색어 입력 (예: a photo of an ocean, a dog)", key="search_input")
    
    if query:
        with st.spinner('수많은 사진 중에서 찾는 중...'):
            try:
                inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
                
                with torch.no_grad():
                    text_outputs = model.get_text_features(**inputs)
                    
                    # [포장지 벗기기 방어 코드]
                    if isinstance(text_outputs, torch.Tensor):
                        text_tensor = text_outputs
                    elif hasattr(text_outputs, 'text_embeds'):
                        text_tensor = text_outputs.text_embeds
                    elif hasattr(text_outputs, 'pooler_output'):
                        text_tensor = text_outputs.pooler_output
                    else:
                        text_tensor = text_outputs[0]
                    
                    # 🌟 알맹이에 L2 정규화 적용
                    text_tensor = text_tensor / text_tensor.norm(p=2, dim=-1, keepdim=True)
                    
                    # 512차원 리스트로 깔끔하게 변환 (초과분은 자름)
                    query_vector = text_tensor.flatten().cpu().tolist()[:512]
                
                # DB 검색 (유사도 0.1 이상)
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
                                img = Image.open(result['file_path'])
                                st.image(img, use_container_width=True)
                                st.caption(f"유사도: {result['similarity']:.4f} | {result['file_name']}")
                            except Exception as e:
                                st.error(f"이미지 로드 실패: {result['file_name']}")
                else:
                    st.warning("⚠️ 비슷한 사진을 찾지 못했습니다. 검색어를 바꿔보세요!")
            except Exception as e:
                st.error(f"❌ 검색 중 에러 발생: {e}")

# ------------------------------------------
# [탭 2] 업로드 기능
# ------------------------------------------
with tab_upload:
    st.subheader("새로운 사진을 추가해 AI가 찾을 수 있게 만듭니다")
    uploaded_file = st.file_uploader("이미지 파일 선택", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, width=300)
        
        if st.button("🚀 AI 분석 및 DB 저장"):
            with st.spinner("AI가 이미지를 분석하고 있습니다..."):
                try:
                    save_dir = "demo_images"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    file_path = os.path.join(save_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    img = Image.open(file_path).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        img_outputs = model.get_image_features(**inputs)
                        
                        # [포장지 벗기기 방어 코드]
                        if isinstance(img_outputs, torch.Tensor):
                            img_tensor = img_outputs
                        elif hasattr(img_outputs, 'image_embeds'):
                            img_tensor = img_outputs.image_embeds
                        elif hasattr(img_outputs, 'pooler_output'):
                            img_tensor = img_outputs.pooler_output
                        else:
                            img_tensor = img_outputs[0]
                        
                        # 🌟 알맹이에 L2 정규화 적용
                        img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)
                        
                        # 512차원 리스트로 깔끔하게 변환
                        vector_list = img_tensor.flatten().cpu().tolist()[:512]
                    
                    if len(vector_list) != 512:
                        st.error(f"❌ 데이터 크기 오류! 512차원이어야 합니다. (현재: {len(vector_list)})")
                        st.stop()
                    
                    insert_data = {
                        "file_name": uploaded_file.name,
                        "file_path": os.path.abspath(file_path),
                        "embedding": vector_list
                    }
                    supabase.table("image_embeddings").insert(insert_data).execute()
                    
                    st.success("✅ DB 저장 완료! 이제 검색 탭에서 이 사진을 찾을 수 있습니다.")
                except Exception as e:
                    st.error(f"❌ 처리 중 에러 발생: {e}")
