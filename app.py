import streamlit as st

# ==========================================
# 1. 웹 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="나만의 AI 갤러리",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 멀티모달 AI 사진 검색 엔진")
st.markdown("자연어로 사진을 검색하고, 새로운 사진을 클라우드에 업로드해보세요.")

# ==========================================
# 2. 화면을 2개의 탭으로 나누기
# ==========================================
tab_search, tab_upload = st.tabs(["🔍 사진 검색하기", "☁️ 새 사진 업로드"])

# ------------------------------------------
# [탭 1] 검색 화면
# ------------------------------------------
with tab_search:
    st.subheader("어떤 사진을 찾으시나요?")
    
    # 검색어 입력창 (엔터를 치면 아래 로직이 실행됨)
    query = st.text_input("검색어를 영어로 입력하세요 (예: a photo of a dog, a receipt)", key="search_input")
    
    if query:
        st.info(f"'{query}'에 대한 검색을 시작합니다... (AI 연결 예정)")
        
        # 임시 UI: 검색 결과가 나왔다고 가정하고 이미지 틀 보여주기
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("https://via.placeholder.com/300x200?text=Result+1", caption="1위 (유사도: 0.85)")
        with col2:
            st.image("https://via.placeholder.com/300x200?text=Result+2", caption="2위 (유사도: 0.72)")
        with col3:
            st.image("https://via.placeholder.com/300x200?text=Result+3", caption="3위 (유사도: 0.61)")

# ------------------------------------------
# [탭 2] 업로드 화면
# ------------------------------------------
with tab_upload:
    st.subheader("새로운 사진을 AI 장부에 추가하기")
    
    # 파일 업로드 버튼
    uploaded_file = st.file_uploader("여기에 이미지를 드래그하거나 클릭해서 업로드하세요.", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        st.success("파일이 성공적으로 선택되었습니다!")
        st.image(uploaded_file, width=300)
        
        if st.button("AI 분석 및 클라우드(Supabase) 저장"):
            st.warning("아직 연결되지 않았습니다! (저장 로직 추가 예정)")
