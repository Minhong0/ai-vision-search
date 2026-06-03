import datetime
import streamlit as st
from database import supabase
from embeddings import get_text_embedding
from cards import render_search_card

TAG_BONUS = 0.08


def render():
    st.subheader("🔍 텍스트 검색")
    st.caption("CLIP AI 유사도 + 태그 텍스트 일치 점수를 합산하여 결과를 보여줍니다.")

    if st.session_state.tag_click_query:
        st.session_state.search_input = st.session_state.tag_click_query
        st.session_state.tag_click_query = ""

    query = st.text_input("검색어", placeholder="예: 안전모 쓴 작업자, 불량 부품", key="search_input")

    p1, p2 = st.columns(2)
    with p1:
        threshold = st.slider("유사도 커트라인", 0.0, 0.4, 0.18, 0.01, key="threshold_t")
    with p2:
        count = st.number_input("최대 개수", min_value=1, max_value=50, value=15, key="count_t")

    with st.container(border=True):
        st.markdown("**상세 필터**")
        c1, c2 = st.columns(2)
        with c1:
            use_date = st.checkbox("📅 날짜 필터", key="date_chk_t")
            if use_date:
                dc1, dc2 = st.columns(2)
                with dc1:
                    start_date = st.date_input(
                        "시작일", datetime.date.today() - datetime.timedelta(days=30), key="sd_t"
                    )
                with dc2:
                    end_date = st.date_input("종료일", datetime.date.today(), key="ed_t")
            else:
                start_date = end_date = None
        with c2:
            use_size = st.checkbox("💾 용량 필터", key="size_chk_t")
            if use_size:
                min_mb = st.number_input("최소 용량 (MB)", 0.0, 100.0, 1.0, 0.5, key="size_t")
                min_kb = int(min_mb * 1024)
            else:
                min_kb = None

    sds = start_date.strftime("%Y-%m-%d") if start_date else None
    eds = end_date.strftime("%Y-%m-%d") if end_date else None

    if not query:
        st.info("검색어를 입력하면 결과가 갤러리로 표시됩니다.")
        return

    if query != st.session_state.last_query:
        st.session_state.display_count = 5
        st.session_state.last_query = query

    with st.spinner("AI 분석 중..."):
        try:
            query_vector = get_text_embedding(query)

            clip_res = supabase.rpc("match_images", {
                "query_embedding": query_vector,
                "match_threshold": threshold,
                "match_count": count,
                "filter_start_date": sds,
                "filter_end_date": eds,
                "filter_min_size_kb": min_kb,
            }).execute()

            merged = {}
            for r in (clip_res.data or []):
                r["clip_score"] = r["similarity"]
                r["tag_score"] = 0.0
                merged[r["id"]] = r

            tq = supabase.table("image_embeddings").select(
                "id, file_name, file_path, file_size_kb, tags"
            ).ilike("tags", f"%{query}%")
            if sds:
                tq = tq.gte("created_at", sds)
            if eds:
                tq = tq.lte("created_at", eds + "T23:59:59")
            if min_kb:
                tq = tq.gte("file_size_kb", min_kb)
            tag_res = tq.execute()

            for r in (tag_res.data or []):
                if r["id"] in merged:
                    merged[r["id"]]["tag_score"] = TAG_BONUS
                    merged[r["id"]]["tags"] = r.get("tags", "")
                else:
                    r["clip_score"] = 0.0
                    r["tag_score"] = TAG_BONUS
                    merged[r["id"]] = r

            for r in merged.values():
                r["similarity"] = round(r["clip_score"] + r["tag_score"], 4)

            results = sorted(merged.values(), key=lambda x: x["similarity"], reverse=True)
            results = [r for r in results if r["similarity"] >= threshold][:count]

            if results:
                st.success(f"🎉 총 {len(results)}장의 사진을 찾았습니다!")
                displayed = results[:st.session_state.display_count]
                for s in range(0, len(displayed), 5):
                    cols = st.columns(5)
                    for col, result in zip(cols, displayed[s:s + 5]):
                        with col:
                            render_search_card(result)
                if st.session_state.display_count < len(results):
                    if st.button("더 보기", use_container_width=True, key="more_t"):
                        st.session_state.display_count += 5
                        st.rerun()
            else:
                st.warning("⚠️ 사진을 찾지 못했습니다. 커트라인 수치를 낮춰보세요!")
        except Exception as e:
            st.error(f"❌ 검색 중 에러 발생: {e}")
