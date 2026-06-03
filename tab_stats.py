import collections
import streamlit as st
from database import fetch_stats


def render():
    st.subheader("📊 갤러리 통계")
    if st.button("🔄 통계 새로고침"):
        fetch_stats.clear()
        st.rerun()

    try:
        stat_records = fetch_stats()

        if not stat_records:
            st.info("저장된 사진이 없습니다.")
            return

        total_count = len(stat_records)
        total_size_mb = sum(r.get("file_size_kb", 0) for r in stat_records) / 1024
        tagged_count = sum(1 for r in stat_records if r.get("tags"))

        m1, m2, m3 = st.columns(3)
        m1.metric("총 이미지 수", f"{total_count}장")
        m2.metric("총 용량", f"{total_size_mb:.1f} MB")
        m3.metric("태그 있는 이미지", f"{tagged_count}장")

        st.divider()

        tag_counter = collections.Counter()
        for r in stat_records:
            for tag in [t.strip() for t in (r.get("tags") or "").split(",") if t.strip()]:
                tag_counter[tag] += 1

        if tag_counter:
            st.markdown("**🏷️ 태그별 이미지 수**")
            st.bar_chart(dict(sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)[:20]))
        else:
            st.info("태그가 있는 이미지가 없습니다.")

        st.divider()

        st.markdown("**📅 날짜별 업로드 추이**")
        date_counter = collections.Counter()
        for r in stat_records:
            d = (r.get("created_at") or "")[:10]
            if d:
                date_counter[d] += 1
        if date_counter:
            st.line_chart(dict(sorted(date_counter.items())))

    except Exception as e:
        st.error(f"통계 로딩 에러: {e}")
