import uuid
import streamlit as st
import tab_search
import tab_image
import tab_upload
import tab_manage
import tab_stats

st.set_page_config(
    page_title="인제 클라우드 갤러리",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; text-align: center; }
    .subtitle { text-align: center; color: #888; font-size: 0.95rem; margin-bottom: 1.8rem; }
    [data-testid="stImage"] img {
        aspect-ratio: 1 / 1 !important;
        object-fit: cover !important;
        border-radius: 8px !important;
    }
    [data-testid="stPopover"] > button {
        background-color: transparent !important;
        border: none !important;
        color: #888 !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        padding: 0 !important;
        min-height: 0 !important;
        margin-top: -5px !important;
    }
    [data-testid="stPopover"] > button:hover {
        color: #000 !important;
        background-color: transparent !important;
    }
    div[data-stale="true"] { opacity: 1 !important; filter: none !important; transition: none !important; }
</style>
    """,
    unsafe_allow_html=True,
)

# ── 세션 상태 초기화 ────────────────────────────────────────────────
if "display_count" not in st.session_state:
    st.session_state.display_count = 5
if "display_count_img" not in st.session_state:
    st.session_state.display_count_img = 5
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())
if "img_search_ref_id" not in st.session_state:
    st.session_state.img_search_ref_id = None
if "img_search_ref_url" not in st.session_state:
    st.session_state.img_search_ref_url = None
if "tag_click_query" not in st.session_state:
    st.session_state.tag_click_query = ""
if "manage_page" not in st.session_state:
    st.session_state.manage_page = 0

# ── 헤더 ────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 자연어 클라우드 갤러리</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI 유사도 + 태그 검색 · 클라우드 업로드 · 갤러리 관리</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.info("AI 모델: Bingsu/clip-vit-large-patch14-ko")

# ── 탭 구성 ─────────────────────────────────────────────────────────
t_text, t_img, t_upload, t_manage, t_stats = st.tabs([
    "🔍 텍스트 검색", "🖼️ 이미지 검색", "☁️ 업로드", "🗂️ 관리", "📊 통계"
])

with t_text:
    tab_search.render()

with t_img:
    tab_image.render()

with t_upload:
    tab_upload.render()

with t_manage:
    tab_manage.render()

with t_stats:
    tab_stats.render()
