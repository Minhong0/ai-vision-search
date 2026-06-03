import uuid
import streamlit as st
from database import fetch_stats
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
    /* ── 1. Google Font ───────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;800&display=swap');
    html, body, [class*="st-"], .stMarkdown, .stText, label {
        font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── 2. 이미지 카드 ────────────────────────────────────────── */
    [data-testid="stImage"] img {
        aspect-ratio: 1 / 1 !important;
        object-fit: cover !important;
        border-radius: 10px !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    }
    [data-testid="stImage"]:hover img {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.18) !important;
    }

    /* ── 3. 탭 바 ─────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 8px 16px !important;
        transition: background 0.15s;
        color: #6b7280 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(79, 70, 229, 0.06) !important;
        color: #4f46e5 !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(79, 70, 229, 0.1) !important;
        color: #4f46e5 !important;
        border-bottom: 2px solid #4f46e5 !important;
    }

    /* ── 4. 태그 버튼 → pill ─────────────────────────────────── */
    button[data-testid="baseButton-secondary"] {
        border-radius: 999px !important;
        padding: 2px 12px !important;
        font-size: 0.78rem !important;
        min-height: 28px !important;
        line-height: 1.3 !important;
        border: 1px solid #d1d5db !important;
        background: #f9fafb !important;
        color: #374151 !important;
        transition: all 0.15s !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background: #ede9fe !important;
        border-color: #7c3aed !important;
        color: #4f46e5 !important;
    }
    /* full-width 버튼(더 보기, 이전/다음 등)은 pill 제외 */
    button[data-testid="baseButton-secondary"][style*="width: 100%"],
    button[data-testid="baseButton-secondary"][style*="width:100%"] {
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-size: 0.9rem !important;
        min-height: 38px !important;
    }

    /* ── 5. 팝오버 버튼 ───────────────────────────────────────── */
    [data-testid="stPopover"] > button {
        background-color: transparent !important;
        border: none !important;
        color: #9ca3af !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        padding: 0 !important;
        min-height: 0 !important;
        margin-top: -5px !important;
        border-radius: 4px !important;
    }
    [data-testid="stPopover"] > button:hover {
        color: #111827 !important;
        background-color: #f3f4f6 !important;
    }

    /* ── 6. 이미지 전체화면 버튼(expand_more) 숨김 ─────────────── */
    [data-testid="StyledFullScreenButton"] {
        display: none !important;
    }

    /* ── 7. 깜빡임 방지 ───────────────────────────────────────── */
    div[data-stale="true"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }
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

# ── 헤더 (그라디언트 카드) ───────────────────────────────────────────
st.markdown(
    """
<div style="
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    padding: 2rem 1.5rem 1.8rem;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 24px rgba(79,70,229,0.25);
">
    <h1 style="color:white; margin:0; font-size:2rem; font-weight:800; letter-spacing:-0.5px;">
        🔍 자연어 클라우드 갤러리
    </h1>
    <p style="color:rgba(255,255,255,0.82); margin:0.5rem 0 0; font-size:0.92rem;">
        AI 유사도 + 태그 검색 &nbsp;·&nbsp; 클라우드 업로드 &nbsp;·&nbsp; 갤러리 관리
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── 사이드바 ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 AI 모델")
    st.code("clip-vit-large-patch14-ko", language=None)
    st.divider()
    st.markdown("### 📊 현황")
    try:
        _stats = fetch_stats()
        _total = len(_stats)
        _tagged = sum(1 for r in _stats if r.get("tags"))
        c1, c2 = st.columns(2)
        c1.metric("이미지", f"{_total}장")
        c2.metric("태그 지정", f"{_tagged}장")
    except Exception:
        st.caption("통계를 불러오는 중...")

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
