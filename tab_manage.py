import streamlit as st
from database import fetch_gallery
from cards import render_manage_card

MANAGE_PER_PAGE = 20


def render():
    st.subheader("🗂️ 이미지 관리")

    top1, top2 = st.columns([1, 1])
    with top1:
        if st.button("🔄 새로고침", use_container_width=True):
            fetch_gallery.clear()
            st.session_state.manage_page = 0
            st.rerun()
    with top2:
        cols_n = st.selectbox("한 줄 수", [3, 4, 5, 6, 7], index=2)

    try:
        records = fetch_gallery()

        if not records:
            st.info("현재 저장된 사진이 없습니다.")
            return

        total = len(records)
        page = st.session_state.manage_page
        total_pages = (total + MANAGE_PER_PAGE - 1) // MANAGE_PER_PAGE
        page = min(page, total_pages - 1)

        start_idx = page * MANAGE_PER_PAGE
        page_records = records[start_idx:start_idx + MANAGE_PER_PAGE]

        st.caption(f"전체 {total}장 중 {start_idx + 1}~{min(start_idx + MANAGE_PER_PAGE, total)}장 표시")

        for s in range(0, len(page_records), cols_n):
            cols = st.columns(cols_n)
            for col, record in zip(cols, page_records[s:s + cols_n]):
                with col:
                    render_manage_card(record)

        if total_pages > 1:
            pg1, pg2, pg3 = st.columns([1, 3, 1])
            with pg1:
                if page > 0:
                    if st.button("◀ 이전", use_container_width=True):
                        st.session_state.manage_page -= 1
                        st.rerun()
            with pg2:
                st.markdown(
                    f"<div style='text-align:center;padding-top:8px'>{page + 1} / {total_pages}</div>",
                    unsafe_allow_html=True,
                )
            with pg3:
                if page < total_pages - 1:
                    if st.button("다음 ▶", use_container_width=True):
                        st.session_state.manage_page += 1
                        st.rerun()

    except Exception as e:
        st.error(f"에러: {e}")
