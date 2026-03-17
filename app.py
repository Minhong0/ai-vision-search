import streamlit as st
import torch
from transformers import CLIPModel, CLIPProcessor
from supabase import create_client, Client

# ==========================================
# 1. 환경 설정 (Supabase 연동)
# ==========================================
SUPABASE_URL = "https://[본인의_프로젝트_ID].supabase.co"
SUPABASE_KEY = "본인의_anon_public_key"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 2. AI 모델 로드 (★ 핵심: 캐싱으로 속도 최적화)
# ==========================================
# @st.cache_resource를 달아주면, 검색 버튼을 누를 때마다 모델을 다시 다운받는 대참사를 막아줍니다.
@st.cache_resource
def load_ai_model():
    model_id = "Bingsu/clip-vit-base-patch32-ko"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval() # 추론 모드
    return model, processor

model, processor = load_ai_model()

# ==========================================
# 3. Streamlit 웹 화면 UI 구성
# ==========================================
st.set_page_config(page_title="Vision RAG 갤러리", layout="wide")
st.title("🔍 폐쇄망 환경을 위한 지능형 이미지 검색 시스템 (PoC)")
st.markdown("파일명이 아닌 **'자연어(의미)'**로 사내 비정형 데이터를 검색해보세요.")

# 검색 설정 바 (사이드바)
with st.sidebar:
    st.header("⚙️ 검색 설정 (Threshold)")
    match_threshold = st.slider("최소 일치율 커트라인 (오탐지 방지)", 0.0, 0.5, 0.25, 0.01)
    match_count = st.slider("검색 결과 출력 개수", 1, 20, 5)

# 메인 검색창
search_query = st.text_input("무엇을 찾으시나요?", placeholder="예: 초원 위에 있는 건물, 위장막을 친 군용 차량")

# ==========================================
# 4. 검색 실행 및 결과 출력 파이프라인
# ==========================================
if st.button("🚀 검색 시작"):
    if not search_query:
        st.warning("검색어를 입력해주세요!")
    else:
        with st.spinner('AI가 수천 장의 이미지를 분석 중입니다...'):
            try:
                # [Step A] 사용자의 한국어 검색어를 512차원 벡터로 변환
                inputs = processor(text=[search_query], return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    text_features = model.get_text_features(**inputs)
                    # 필수: DB에 있는 이미지 벡터와 비교하기 위해 똑같이 L2 정규화
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                    query_embedding = text_features.squeeze().tolist()

                # [Step B] Supabase 내부의 RPC 함수(match_images) 호출!
                response = supabase.rpc(
                    'match_images', 
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': match_threshold,
                        'match_count': match_count
                    }
                ).execute()

                results = response.data

                # [Step C] 검색 결과 화면에 뿌려주기
                if not results:
                    st.error(f"'{search_query}'에 대한 검색 결과가 없습니다. (커트라인을 낮춰보세요)")
                else:
                    st.success(f"총 {len(results)}개의 일치하는 데이터를 찾았습니다!")
                    
                    # 결과를 예쁘게 갤러리 형태로 배치 (Streamlit columns 활용)
                    cols = st.columns(3)
                    for idx, item in enumerate(results):
                        with cols[idx % 3]:
                            # Supabase에서 가져온 이미지 URL 띄우기
                            st.image(item['file_path'], use_container_width=True)
                            # 유사도 점수와 파일명 표기
                            st.caption(f"📁 {item['file_name']} (유사도: {item['similarity']:.3f})")

            except Exception as e:
                st.error(f"검색 중 오류가 발생했습니다: {e}")
