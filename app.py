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
# 2. AI 모델 및 DB 연결 (캐싱 적용: 최초 1회만 로드)
# ==========================================
@st.cache_resource
def load_system():
    # secrets.toml 또는 Streamlit Cloud Secrets에서 안전하게 키를 불러옵니다.
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    sb = create_client(url, key)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    
    # 모델 로드
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
                # 1. 텍스트를 AI 벡터로 변환
                inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
                
                with torch.no_grad():
                    text_features = model.get_text_features(**inputs)
                    
                    # 🌟 [핵심] L2 정규화: AI의 감정 크기를 일정하게 맞춰 정확한 비교 가능하게 함
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                    
                    # 512차원 리스트로 깔끔하게 변환 (JSON 에러 완벽 방어)
                    query_vector = text_features.flatten().cpu().tolist()
                
                # 2. Supabase DB에서 유사도 검색
                response = supabase.rpc("match_images", {
                    "query_embedding": query_vector,
                    "match_threshold": 0.1,  # 정규화를 거쳤으므로 0.1~0.25 사이가 적당합니다
                    "match_count": 3
                }).execute()
                
                results = response.data
                
                # 3. 결과 화면에 뿌려주기
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
                    # 1. 업로드된 파일을 로컬 폴더에 먼저 임시 저장
                    save_dir = "demo_images"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    file_path = os.path.join(save_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # 2. 저장된 이미지로 벡터 추출
                    img = Image.open(file_path).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        img_features = model.get_image_features(**inputs)
                        
                        # 🌟 [핵심] L2 정규화 적용 (텍스트와 동일한 기준으로 맞춤)
                        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
                        
                        # 512차원 리스트로 깔끔하게 변환
                        vector_list = img_features.flatten().cpu().tolist()
                    
                    # 💡 DB에 넣기 전 마지막 검문소
                    if len(vector_list) != 512:
                        st.error(f"❌ 데이터 크기 오류! 512차원이어야 합니다. (현재: {len(vector_list)})")
                        st.stop()
                    
                    # 3. Supabase DB에 Insert
                    insert_data = {
                        "file_name": uploaded_file.name,
                        "file_path": os.path.abspath(file_path),
                        "embedding": vector_list
                    }
                    supabase.table("image_embeddings").insert(insert_data).execute()
                    
                    st.success("✅ DB 저장 완료! 이제 검색 탭에서 이 사진을 찾을 수 있습니다.")
                except Exception as e:
                    st.error(f"❌ 처리 중 에러 발생: {e}")
