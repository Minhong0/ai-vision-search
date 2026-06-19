import io
import os
import uuid
import requests
import streamlit as st
from PIL import Image
from tab_upload import _save_image
from database import fetch_gallery, fetch_stats

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT = 8


def _get_secret(key: str) -> str | None:
    """Try Streamlit secrets first (local dev), then fall back to environment variables.

    This allows:
    - Local dev: .streamlit/secrets.toml -> st.secrets["KEY"]
    - CI / deployment: environment variables (exposed from GitHub Actions secrets)
    """
    try:
        if hasattr(st, "secrets"):
            val = st.secrets.get(key)
            if val:
                return val
    except Exception:
        pass
    return os.environ.get(key)


def _search_naver(query: str, display: int) -> list:
    client_id = _get_secret("NAVER_CLIENT_ID")
    client_secret = _get_secret("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("Naver API keys are not configured")

    resp = requests.get(
        "https://openapi.naver.com/v1/search/image",
        headers={
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        },
        params={"query": query, "display": display, "sort": "sim"},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return [
        {"title": item.get("title", ""), "url": item["link"], "thumb": item["thumbnail"]}
        for item in resp.json().get("items", [])
    ]


def _search_google(query: str, display: int) -> list:
    api_key = _get_secret("GOOGLE_API_KEY")
    cse_id = _get_secret("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        raise RuntimeError("Google Custom Search keys are not configured")

    resp = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "searchType": "image",
            "num": min(display, 10),
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return [
        {
            "title": item.get("title", ""),
            "url": item["link"],
            "thumb": item.get("image", {}).get("thumbnailLink", item["link"]),
        }
        for item in resp.json().get("items", [])
    ]


def _download_image(url: str, thumb_url: str) -> Image.Image | None:
    for target in [url, thumb_url]:
        try:
            r = requests.get(target, headers=_HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            continue
    return None


def render():
    st.subheader("🌐 웹 이미지 검색 & 저장")
    st.caption("네이버/구글에서 이미지를 검색하고 선택한 이미지를 바로 갤러리에 저장합니다.")

    # ── 소스 선택 ──────────────────────────────────────────────────────
    naver_ok = bool(_get_secret("NAVER_CLIENT_ID") and _get_secret("NAVER_CLIENT_SECRET"))
    google_ok = bool(_get_secret("GOOGLE_API_KEY") and _get_secret("GOOGLE_CSE_ID"))

    if not naver_ok and not google_ok:
        st.error(
            "API 키가 없습니다. `.streamlit/secrets.toml` 에 다음 중 하나를 추가하거나,\n"
            "GitHub Actions 또는 배포 환경의 환경변수로 `NAVER_CLIENT_ID`/`NAVER_CLIENT_SECRET` 또는 `GOOGLE_API_KEY`/`GOOGLE_CSE_ID` 를 설정하세요.\n\n"
            "**Naver (권장, 무료 25,000회/일)**\n"
            "```toml\nNAVER_CLIENT_ID = \"...\"\nNAVER_CLIENT_SECRET = \"...\"\n```\n"
            "발급: https://developers.naver.com → 애플리케이션 등록 → 검색 API"
        )
        return

    sources = []
    if naver_ok:
        sources.append("Naver")
    if google_ok:
        sources.append("Google")

    source = st.radio("검색 소스", sources, horizontal=True)

    # ── 검색 입력 ──────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("검색어", placeholder="예: 공장 안전모 작업자")
    with col2:
        display = st.number_input("결과 수", min_value=5, max_value=30, value=15)

    if not query:
        st.info("검색어를 입력하면 이미지 목록이 표시됩니다.")
        return

    if "import_results" not in st.session_state or st.session_state.get("import_query") != (query, source):
        if st.button("🔍 검색", type="primary", use_container_width=True):
            with st.spinner(f"{source}에서 '{query}' 검색 중..."):
                try:
                    items = _search_naver(query, display) if source == "Naver" else _search_google(query, display)
                    st.session_state.import_results = items
                    st.session_state.import_query = (query, source)
                    st.rerun()
                except Exception as e:
                    st.error(f"검색 오류: {e}")
        return

    items = st.session_state.import_results
    if not items:
        st.warning("검색 결과가 없습니다.")
        return

    st.success(f"**{len(items)}장** 검색됨 — 저장할 이미지를 선택하세요.")
    st.caption("⚠️ 검색 결과 이미지의 저작권은 원저작자에게 있습니다.")

    # ── 썸네일 그리드 + 체크박스 ────────────────────────────────────────
    selected_indices = []
    for row_start in range(0, len(items), 5):
        cols = st.columns(5)
        for col, idx in zip(cols, range(row_start, min(row_start + 5, len(items)))):
            item = items[idx]
            with col:
                try:
                    st.image(item["thumb"], use_container_width=True)
                except Exception:
                    st.caption("미리보기 없음")
                if st.checkbox("선택", key=f"import_{idx}"):
                    selected_indices.append(idx)

    st.divider()

    # ── 태그 + 저장 ────────────────────────────────────────────────────
    tags = st.text_input("태그 (선택사항)", placeholder="예: 안전모, 작업자")

    if not selected_indices:
        st.info(f"이미지를 선택하면 저장 버튼이 활성화됩니다. (현재 0장 선택)")
        return

    if st.button(f"💾 선택한 이미지 저장 ({len(selected_indices)}장)", type="primary", use_container_width=True):
        success, fail = 0, 0
        with st.spinner("다운로드 및 업로드 중..."):
            for idx in selected_indices:
                item = items[idx]
                img = _download_image(item["url"], item["thumb"])
                if img is None:
                    st.warning(f"❌ 다운로드 실패: {item['title'] or item['url']}")
                    fail += 1
                    continue
                try:
                    title = item["title"] or f"web_{uuid.uuid4().hex[:8]}"
                    safe_name = "".join(c for c in title if c.isalnum() or c in " _-")[:40].strip()
                    filename = f"{safe_name or uuid.uuid4().hex[:8]}.jpg"
                    _save_image(img, filename, tags or None, use_ocr=False)
                    success += 1
                except Exception as e:
                    st.warning(f"❌ 저장 실패 ({item['title']}): {e}")
                    fail += 1

        fetch_gallery.clear()
        fetch_stats.clear()
        if success:
            st.success(f"✅ {success}장 저장 완료!" + (f" ({fail}장 실패)" if fail else ""))
        else:
            st.error("저장에 실패했습니다.")
        del st.session_state.import_results
        del st.session_state.import_query
        st.rerun()
