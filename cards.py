import time
import streamlit as st
from database import supabase, fetch_gallery, fetch_stats


def render_search_card(result: dict):
    try:
        st.image(result["file_path"], use_container_width=True)

        col_title, col_menu = st.columns([5, 1])
        with col_title:
            st.markdown(f"**{result['file_name']}**")
            for tag in [t.strip() for t in (result.get("tags") or "").split(",") if t.strip()]:
                if st.button(f"🏷️ {tag}", key=f"tag_src_{result['id']}_{tag}"):
                    st.session_state.tag_click_query = tag
                    st.rerun()

        with col_menu:
            with st.popover("⋮"):
                raw_size = result.get("file_size_kb", 0)
                clip_s = result.get("clip_score", result.get("similarity", 0))
                tag_s = result.get("tag_score", 0.0)
                st.caption(
                    f"🎯 합산 {result['similarity']:.3f} "
                    f"(AI {clip_s:.3f} + 태그 {tag_s:.2f}) · 💾 {int(raw_size)}KB"
                )

                new_name = st.text_input("이름 변경", value=result["file_name"], key=f"rn_src_{result['id']}")
                if st.button("💾 저장", key=f"rn_btn_src_{result['id']}", use_container_width=True):
                    supabase.table("image_embeddings").update({"file_name": new_name}).eq("id", result["id"]).execute()
                    fetch_gallery.clear()
                    st.toast("변경 완료!")
                    time.sleep(0.5)
                    st.rerun()

                st.divider()

                if st.button("🔍 비슷한 사진 찾기", key=f"sim_src_{result['id']}", use_container_width=True):
                    st.session_state.img_search_ref_id = result["id"]
                    st.session_state.img_search_ref_url = result["file_path"]
                    st.session_state.display_count_img = 5
                    st.rerun()

                st.divider()

                st.link_button("📥 다운로드", result["file_path"], use_container_width=True)

                if st.button("🗑️ 삭제", key=f"del_src_{result['id']}", use_container_width=True, type="primary"):
                    supabase.storage.from_("images").remove([result["file_path"].split("/")[-1]])
                    supabase.table("image_embeddings").delete().eq("id", result["id"]).execute()
                    fetch_gallery.clear()
                    fetch_stats.clear()
                    st.toast("삭제 완료!")
                    time.sleep(0.5)
                    st.rerun()
    except Exception:
        st.error("이미지 에러")


def render_manage_card(record: dict):
    st.image(record["file_path"], use_container_width=True)

    col_title, col_menu = st.columns([5, 1])
    with col_title:
        st.markdown(f"**{record['file_name']}**")
        for tag in [t.strip() for t in (record.get("tags") or "").split(",") if t.strip()]:
            if st.button(f"🏷️ {tag}", key=f"tag_mng_{record['id']}_{tag}"):
                st.session_state.tag_click_query = tag
                st.rerun()

    with col_menu:
        with st.popover("⋮"):
            raw_size = record.get("file_size_kb", 0)
            created_date = record.get("created_at", "최근")[:10]
            st.caption(f"📅 {created_date} · 💾 {int(raw_size)}KB")

            new_name = st.text_input("이름 변경", value=record["file_name"], key=f"rn_mng_{record['id']}")
            if st.button("💾 저장", key=f"rn_btn_mng_{record['id']}", use_container_width=True):
                supabase.table("image_embeddings").update({"file_name": new_name}).eq("id", record["id"]).execute()
                fetch_gallery.clear()
                st.toast("변경 완료!")
                time.sleep(0.5)
                st.rerun()

            st.divider()

            current_tags = record.get("tags") or ""
            new_tags = st.text_input(
                "🏷️ 태그 편집", value=current_tags,
                key=f"tag_edit_mng_{record['id']}",
                placeholder="쉼표로 구분: 안전모, 불량 부품",
            )
            if st.button("🏷️ 태그 저장", key=f"tag_save_mng_{record['id']}", use_container_width=True):
                supabase.table("image_embeddings").update({"tags": new_tags or None}).eq("id", record["id"]).execute()
                fetch_gallery.clear()
                fetch_stats.clear()
                st.toast("태그 저장 완료!")
                time.sleep(0.5)
                st.rerun()

            st.divider()

            if st.button("🔍 비슷한 사진 찾기", key=f"sim_mng_{record['id']}", use_container_width=True):
                st.session_state.img_search_ref_id = record["id"]
                st.session_state.img_search_ref_url = record["file_path"]
                st.session_state.display_count_img = 5
                st.toast("이미지 검색 탭에서 유사 사진을 확인하세요!", icon="🔍")
                st.rerun()

            st.divider()

            st.link_button("📥 다운로드", record["file_path"], use_container_width=True)

            if st.button("🗑️ 삭제", key=f"del_mng_{record['id']}", use_container_width=True, type="primary"):
                supabase.storage.from_("images").remove([record["file_path"].split("/")[-1]])
                supabase.table("image_embeddings").delete().eq("id", record["id"]).execute()
                fetch_gallery.clear()
                fetch_stats.clear()
                st.toast("삭제 완료!")
                time.sleep(0.5)
                st.rerun()
