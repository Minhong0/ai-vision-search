import datetime
import streamlit as st
from PIL import Image
from database import supabase
from embeddings import get_image_embedding
from cards import render_search_card


def render():
    st.subheader("🖼️ 이미지로 검색")
    st.caption("기준 사진과 시각적으로 비슷한 분위기의 사진을 찾아줍니다.")

    p1, p2 = st.columns(2)
    with p1:
        threshold = st.slider("유사도 커트라인", 0.0, 1.0, 0.6, 0.01, key="threshold_i")
    with p2:
        count = st.number_input("최대 개수", min_value=1, max_value=50, value=15, key="count_i")

    with st.container(border=True):
        st.markdown("**상세 필터**")
        c1, c2 = st.columns(2)
        with c1:
            use_date = st.checkbox("📅 날짜 필터", key="date_chk_i")
            if use_date:
                dc1, dc2 = st.columns(2)
                with dc1:
                    start_date = st.date_input(
                        "시작일", datetime.date.today() - datetime.timedelta(days=30), key="sd_i"
                    )
                with dc2:
                    end_date = st.date_input("종료일", datetime.date.today(), key="ed_i")
            else:
                start_date = end_date = None
        with c2:
            use_size = st.checkbox("💾 용량 필터", key="size_chk_i")
            if use_size:
                min_mb = st.number_input("최소 용량 (MB)", 0.0, 100.0, 1.0, 0.5, key="size_i")
                min_kb = int(min_mb * 1024)
            else:
                min_kb = None

    sds = start_date.strftime("%Y-%m-%d") if start_date else None
    eds = end_date.strftime("%Y-%m-%d") if end_date else None

    query_vector = None

    if st.session_state.img_search_ref_url:
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(st.session_state.img_search_ref_url, caption="기준 사진", use_container_width=True)
        with col_info:
            st.info("관리 탭에서 선택한 사진을 기준으로 유사 사진을 검색합니다.")
            if st.button("❌ 기준 사진 초기화", use_container_width=True):
                st.session_state.img_search_ref_id = None
                st.session_state.img_search_ref_url = None
                st.session_state.display_count_img = 5
                st.rerun()

        emb_res = supabase.table("image_embeddings").select("embedding").eq(
            "id", st.session_state.img_search_ref_id
        ).execute()
        if emb_res.data:
            query_vector = emb_res.data[0]["embedding"]
    else:
        img_file = st.file_uploader(
            "기준 이미지 업로드 (또는 관리 탭에서 '비슷한 사진 찾기' 버튼 사용)",
            type=["png", "jpg", "jpeg"],
            key="img_search_uploader",
        )
        if img_file:
            col_prev, _ = st.columns([1, 2])
            with col_prev:
                st.image(img_file, caption="기준 사진", use_container_width=True)
            with st.spinner("이미지 분석 중..."):
                img = Image.open(img_file).convert("RGB")
                query_vector = get_image_embedding(img)

    if not query_vector:
        st.info("기준 이미지를 업로드하거나, 관리 탭에서 '🔍 비슷한 사진 찾기' 버튼을 눌러주세요.")
        return

    with st.spinner("비슷한 사진 찾는 중..."):
        try:
            response = supabase.rpc("match_images", {
                "query_embedding": query_vector,
                "match_threshold": threshold,
                "match_count": count + 1,
                "filter_start_date": sds,
                "filter_end_date": eds,
                "filter_min_size_kb": min_kb,
            }).execute()

            ref_id = st.session_state.get("img_search_ref_id")
            results = [r for r in (response.data or []) if r["id"] != ref_id][:count]
            for r in results:
                r["clip_score"] = r["similarity"]
                r["tag_score"] = 0.0

            if results:
                st.success(f"🎉 비슷한 분위기의 사진 {len(results)}장을 찾았습니다!")
                displayed = results[:st.session_state.display_count_img]
                for s in range(0, len(displayed), 5):
                    cols = st.columns(5)
                    for col, result in zip(cols, displayed[s:s + 5]):
                        with col:
                            render_search_card(result)
                if st.session_state.display_count_img < len(results):
                    if st.button("더 보기", use_container_width=True, key="more_i"):
                        st.session_state.display_count_img += 5
                        st.rerun()
            else:
                st.warning("⚠️ 비슷한 사진을 찾지 못했습니다. 커트라인 수치를 낮춰보세요!")
        except Exception as e:
            st.error(f"❌ 검색 중 에러 발생: {e}")
