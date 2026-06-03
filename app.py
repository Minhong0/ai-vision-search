import streamlit as st
import torch
import os
import uuid
import requests
import time
import datetime
from PIL import Image
from transformers import AutoProcessor, AutoModel
from supabase import create_client, Client

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="인제 클라우드 갤러리",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# 💡 [CSS 설정] 레이아웃 정상화 & 깔끔한 갤러리 UI
# =====================================================================
st.markdown(
    """
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; text-align: center; }
    .subtitle { text-align: center; color: #888; font-size: 0.95rem; margin-bottom: 1.8rem; }
    
    /* 1. 이미지 1:1 강제 비율 및 둥근 모서리 */
    [data-testid="stImage"] img {
        aspect-ratio: 1 / 1 !important;
        object-fit: cover !important;
        border-radius: 8px !important;
    }
    
    /* 💡 폭탄(gap: 0rem) 제거됨! 스트림릿 기본 여백은 그대로 둡니다. */
    
    /* 2. 팝오버(⋮) 버튼을 제목과 일직선이 되도록 예쁘게 정렬 */
    [data-testid="stPopover"] > button {
        background-color: transparent !important;
        border: none !important;
        color: #888 !important;
        font-size: 1.3rem !important; /* 점 3개 크기 살짝 키움 */
        font-weight: bold !important;
        padding: 0 !important;
        min-height: 0 !important;
        margin-top: -5px !important; /* 제목(텍스트) 높이와 맞추기 위해 살짝 위로 올림 */
    }
    [data-testid="stPopover"] > button:hover {
        color: #000 !important;
        background-color: transparent !important;
    }
    
    /* 화면 깜빡임 방지 */
    div[data-stale="true"] { opacity: 1 !important; filter: none !important; transition: none !important; }
</style>
    """,
    unsafe_allow_html=True,
)

if "display_count" not in st.session_state:
    st.session_state.display_count = 5
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())
if "img_search_ref_id" not in st.session_state:
    st.session_state.img_search_ref_id = None
if "img_search_ref_url" not in st.session_state:
    st.session_state.img_search_ref_url = None


@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase = init_supabase()


@st.cache_resource(show_spinner="모델 로딩중...")
def load_ai_model():
    processor = AutoProcessor.from_pretrained("Bingsu/clip-vit-large-patch14-ko")
    model = AutoModel.from_pretrained("Bingsu/clip-vit-large-patch14-ko").to(device)
    return processor, model


processor, model = load_ai_model()


st.markdown('<div class="main-title">🔍 자연어 클라우드 갤러리</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI 유사도 + 태그 검색 · 클라우드 업로드 · 갤러리 관리</div>', unsafe_allow_html=True)

# =====================================================================
# 🎛️ 사이드바 UI
# =====================================================================
with st.sidebar:
    st.info("AI 모델: Bingsu/clip-vit-large-patch14-ko")


# =====================================================================
# 📇 카드 렌더링 함수 (사진 아래 깔끔하게 정보 배치)
# =====================================================================
def render_search_card(result):
    try:
        st.image(result["file_path"], use_container_width=True)
        
        col_title, col_menu = st.columns([5, 1])
        with col_title:
            st.markdown(f"**{result['file_name']}**")
            
        with col_menu:
            with st.popover("⋮"):
                raw_size = result.get("file_size_kb", 0)
                clip_s = result.get("clip_score", result.get("similarity", 0))
                tag_s = result.get("tag_score", 0.0)
                st.caption(f"🎯 합산 {result['similarity']:.3f} (AI {clip_s:.3f} + 태그 {tag_s:.1f}) · 💾 {int(raw_size)}KB")
                
                new_name = st.text_input("이름 변경", value=result["file_name"], key=f"rn_src_{result['id']}")
                if st.button("💾 저장", key=f"rn_btn_src_{result['id']}", use_container_width=True):
                    supabase.table("image_embeddings").update({"file_name": new_name}).eq("id", result["id"]).execute()
                    st.toast("변경 완료!")
                    time.sleep(0.5)
                    st.rerun()
                
                st.divider()
                
                if st.button("🔍 비슷한 사진 찾기", key=f"sim_src_{result['id']}", use_container_width=True):
                    st.session_state.img_search_ref_id = result["id"]
                    st.session_state.img_search_ref_url = result["file_path"]
                    st.session_state.search_mode = "🖼️ 이미지로 검색"
                    st.session_state.display_count = 5
                    st.rerun()

                st.divider()

                img_data = requests.get(result["file_path"]).content
                st.download_button("📥 다운로드", data=img_data, file_name=result["file_name"], mime="image/jpeg", key=f"dl_src_{result['id']}", use_container_width=True)

                if st.button("🗑️ 삭제", key=f"del_src_{result['id']}", use_container_width=True, type="primary"):
                    supabase.storage.from_("images").remove([result["file_path"].split("/")[-1]])
                    supabase.table("image_embeddings").delete().eq("id", result["id"]).execute()
                    st.toast("삭제 완료!")
                    time.sleep(0.5)
                    st.rerun()
    except Exception:
        st.error("이미지 에러")

def render_manage_card(record):
    st.image(record["file_path"], use_container_width=True)
    
    col_title, col_menu = st.columns([5, 1])
    with col_title:
        st.markdown(f"**{record['file_name']}**")
        
    with col_menu:
        with st.popover("⋮"):
            raw_size = record.get("file_size_kb", 0)
            created_date = record.get("created_at", "최근")[:10]
            st.caption(f"📅 {created_date} · 💾 {int(raw_size)}KB")

            new_name = st.text_input("이름 변경", value=record["file_name"], key=f"rn_mng_{record['id']}")
            if st.button("💾 저장", key=f"rn_btn_mng_{record['id']}", use_container_width=True):
                supabase.table("image_embeddings").update({"file_name": new_name}).eq("id", record["id"]).execute()
                st.toast("변경 완료!")
                time.sleep(0.5)
                st.rerun()
            
            st.divider()
            
            if st.button("🔍 비슷한 사진 찾기", key=f"sim_mng_{record['id']}", use_container_width=True):
                st.session_state.img_search_ref_id = record["id"]
                st.session_state.img_search_ref_url = record["file_path"]
                st.session_state.search_mode = "🖼️ 이미지로 검색"
                st.session_state.display_count = 5
                st.toast("검색 탭에서 유사 사진을 확인하세요!", icon="🔍")
                st.rerun()

            st.divider()

            try:
                img_data = requests.get(record["file_path"]).content
                st.download_button("📥 다운로드", data=img_data, file_name=record["file_name"], mime="image/jpeg", key=f"dl_mng_{record['id']}", use_container_width=True)
            except:
                pass

            if st.button("🗑️ 삭제", key=f"del_mng_{record['id']}", use_container_width=True, type="primary"):
                supabase.storage.from_("images").remove([record["file_path"].split("/")[-1]])
                supabase.table("image_embeddings").delete().eq("id", record["id"]).execute()
                st.toast("삭제 완료!")
                time.sleep(0.5)
                st.rerun()

# =====================================================================
# 화면 탭 구성
# =====================================================================
tab_search, tab_upload, tab_manage = st.tabs(["🔍 사진 검색", "☁️ 사진 업로드", "🗂️ 갤러리 관리"])


# [탭 1] 검색 기능
with tab_search:
    st.subheader("🔍 사진 검색")
    st.caption("CLIP AI 유사도 점수와 태그 텍스트 일치 점수를 합산하여 결과를 보여줍니다.")

    search_mode = st.radio(
        "검색 방식",
        ["📝 텍스트 검색", "🖼️ 이미지로 검색"],
        horizontal=True,
        key="search_mode",
    )

    p1, p2 = st.columns(2)
    with p1:
        match_threshold = st.slider("유사도 커트라인", 0.0, 0.4, 0.18, 0.01)
    with p2:
        match_count = st.number_input("최대 개수", min_value=1, max_value=50, value=15)

    with st.container(border=True):
        st.markdown("**상세 필터**")
        col1, col2 = st.columns(2)
        with col1:
            use_date_filter = st.checkbox("📅 업로드 날짜 필터 사용")
            if use_date_filter:
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    start_date = st.date_input("시작일", datetime.date.today() - datetime.timedelta(days=30))
                with d_col2:
                    end_date = st.date_input("종료일", datetime.date.today())
            else:
                start_date = None
                end_date = None
        with col2:
            use_size_filter = st.checkbox("💾 파일 용량 필터 사용")
            if use_size_filter:
                min_size_mb = st.number_input("최소 용량 (MB)", min_value=0.0, max_value=100.0, value=1.0, step=0.5)
                min_size_kb = int(min_size_mb * 1024)
            else:
                min_size_kb = None

    start_date_str = start_date.strftime("%Y-%m-%d") if start_date else None
    end_date_str = end_date.strftime("%Y-%m-%d") if end_date else None

    # ─── 텍스트 검색 ──────────────────────────────────────────────────────
    if search_mode == "📝 텍스트 검색":
        query = st.text_input("검색어", placeholder="예: 안전모 쓴 작업자, 불량 부품", key="search_input")

        if query:
            if query != st.session_state.last_query:
                st.session_state.display_count = 5
                st.session_state.last_query = query

            with st.spinner("AI 분석 중..."):
                try:
                    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        text_outputs = model.get_text_features(**inputs)
                        text_tensor = text_outputs if isinstance(text_outputs, torch.Tensor) else (text_outputs.text_embeds if hasattr(text_outputs, "text_embeds") else (text_outputs.pooler_output if hasattr(text_outputs, "pooler_output") else text_outputs[0]))
                        final_tensor = text_tensor / text_tensor.norm(p=2, dim=-1, keepdim=True)
                    query_vector = final_tensor.flatten().cpu().tolist()[:768]

                    clip_response = supabase.rpc(
                        "match_images",
                        {
                            "query_embedding": query_vector,
                            "match_threshold": match_threshold,
                            "match_count": match_count,
                            "filter_start_date": start_date_str,
                            "filter_end_date": end_date_str,
                            "filter_min_size_kb": min_size_kb,
                        },
                    ).execute()

                    TAG_BONUS = 0.3
                    merged = {}
                    for r in (clip_response.data or []):
                        r["clip_score"] = r["similarity"]
                        r["tag_score"] = 0.0
                        merged[r["id"]] = r

                    tag_query = supabase.table("image_embeddings").select(
                        "id, file_name, file_path, file_size_kb"
                    ).ilike("tags", f"%{query}%")
                    if start_date_str:
                        tag_query = tag_query.gte("created_at", start_date_str)
                    if end_date_str:
                        tag_query = tag_query.lte("created_at", end_date_str + "T23:59:59")
                    if min_size_kb:
                        tag_query = tag_query.gte("file_size_kb", min_size_kb)
                    tag_response = tag_query.execute()

                    for r in (tag_response.data or []):
                        if r["id"] in merged:
                            merged[r["id"]]["tag_score"] = TAG_BONUS
                        else:
                            r["clip_score"] = 0.0
                            r["tag_score"] = TAG_BONUS
                            merged[r["id"]] = r

                    for r in merged.values():
                        r["similarity"] = round(r["clip_score"] + r["tag_score"], 4)

                    results = sorted(merged.values(), key=lambda x: x["similarity"], reverse=True)
                    results = [r for r in results if r["similarity"] >= match_threshold][:match_count]

                    if results:
                        st.success(f"🎉 총 {len(results)}장의 사진을 찾았습니다!")
                        displayed_results = results[:st.session_state.display_count]
                        for s in range(0, len(displayed_results), 5):
                            cols = st.columns(5)
                            for col, result in zip(cols, displayed_results[s:s + 5]):
                                with col:
                                    render_search_card(result)
                        if st.session_state.display_count < len(results):
                            if st.button("더 보기", use_container_width=True):
                                st.session_state.display_count += 5
                                st.rerun()
                    else:
                        st.warning("⚠️ 사진을 찾지 못했습니다. 커트라인 수치를 낮춰보세요!")
                except Exception as e:
                    st.error(f"❌ 검색 중 에러 발생: {e}")
        else:
            st.info("검색어를 입력하면 결과가 갤러리로 표시됩니다.")

    # ─── 이미지로 검색 ────────────────────────────────────────────────────
    else:
        query_vector = None

        if st.session_state.img_search_ref_url:
            col_img, col_info = st.columns([1, 2])
            with col_img:
                st.image(st.session_state.img_search_ref_url, caption="기준 사진", use_container_width=True)
            with col_info:
                st.info("갤러리에서 선택한 사진을 기준으로 유사 사진을 검색합니다.")
                if st.button("❌ 기준 사진 초기화", use_container_width=True):
                    st.session_state.img_search_ref_id = None
                    st.session_state.img_search_ref_url = None
                    st.session_state.display_count = 5
                    st.rerun()

            emb_res = supabase.table("image_embeddings").select("embedding").eq(
                "id", st.session_state.img_search_ref_id
            ).execute()
            if emb_res.data:
                query_vector = emb_res.data[0]["embedding"]

        else:
            img_file = st.file_uploader(
                "기준 이미지 업로드 (또는 갤러리에서 '비슷한 사진 찾기' 버튼 사용)",
                type=["png", "jpg", "jpeg"],
                key="img_search_uploader",
            )
            if img_file:
                col_prev, _ = st.columns([1, 2])
                with col_prev:
                    st.image(img_file, caption="기준 사진", use_container_width=True)
                with st.spinner("이미지 분석 중..."):
                    img = Image.open(img_file).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        img_outputs = model.get_image_features(**inputs)
                        img_tensor = img_outputs if isinstance(img_outputs, torch.Tensor) else (img_outputs.image_embeds if hasattr(img_outputs, "image_embeds") else (img_outputs.pooler_output if hasattr(img_outputs, "pooler_output") else img_outputs[0]))
                        img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)
                        query_vector = img_tensor.flatten().cpu().tolist()[:768]

        if query_vector:
            with st.spinner("비슷한 사진 찾는 중..."):
                try:
                    response = supabase.rpc(
                        "match_images",
                        {
                            "query_embedding": query_vector,
                            "match_threshold": match_threshold,
                            "match_count": match_count + 1,
                            "filter_start_date": start_date_str,
                            "filter_end_date": end_date_str,
                            "filter_min_size_kb": min_size_kb,
                        },
                    ).execute()

                    ref_id = st.session_state.get("img_search_ref_id")
                    results = [r for r in (response.data or []) if r["id"] != ref_id][:match_count]
                    for r in results:
                        r["clip_score"] = r["similarity"]
                        r["tag_score"] = 0.0

                    if results:
                        st.success(f"🎉 비슷한 분위기의 사진 {len(results)}장을 찾았습니다!")
                        displayed = results[:st.session_state.display_count]
                        for s in range(0, len(displayed), 5):
                            cols = st.columns(5)
                            for col, result in zip(cols, displayed[s:s + 5]):
                                with col:
                                    render_search_card(result)
                        if st.session_state.display_count < len(results):
                            if st.button("더 보기", use_container_width=True, key="img_more"):
                                st.session_state.display_count += 5
                                st.rerun()
                    else:
                        st.warning("⚠️ 비슷한 사진을 찾지 못했습니다. 커트라인 수치를 낮춰보세요!")
                except Exception as e:
                    st.error(f"❌ 검색 중 에러 발생: {e}")
        else:
            st.info("기준 이미지를 업로드하거나, 갤러리(관리 탭)에서 '🔍 비슷한 사진 찾기' 버튼을 눌러주세요.")


# [탭 2] 업로드 기능
with tab_upload:
    st.subheader("📤 사진 업로드 ")

    uploaded_files = st.file_uploader(
        "이미지 파일 선택 (여러 장 드래그 앤 드롭 가능)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
    )

    if uploaded_files:
        st.write(f"총 **{len(uploaded_files)}**장의 사진이 선택되었습니다.")

        with st.container(border=True):
            st.markdown("**🏷️ 태그 (선택사항)**")
            uploaded_tags = st.text_input(
                "이 사진들의 특징이나 이름을 입력해주세요 (검색 시 태그 일치 점수로 활용됩니다)",
                placeholder="예: 불량 부품, 안전모, 작업자...",
            )

        for start in range(0, len(uploaded_files), 5):
            cols = st.columns(5)
            chunk = uploaded_files[start:start + 5]
            for col, file in zip(cols, chunk):
                with col:
                    with st.container(border=True):
                        st.image(file, use_container_width=True) 
                        st.caption(file.name)

        btn_save = st.button("💾 저장", use_container_width=True)

        if btn_save:
            with st.spinner("이미지 업로드 및 분석 중..."):
                success_count = 0
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        original_filename = uploaded_file.name
                        ext = os.path.splitext(original_filename)[1]
                        safe_filename = f"{uuid.uuid4().hex}{ext}"

                        file_bytes = uploaded_file.getvalue()
                        file_size_kb = len(file_bytes) // 1024

                        supabase.storage.from_("images").upload(
                            path=safe_filename,
                            file=file_bytes,
                            file_options={"content-type": uploaded_file.type},
                        )

                        public_url = supabase.storage.from_("images").get_public_url(safe_filename)

                        img = Image.open(uploaded_file).convert("RGB")
                        inputs = processor(images=img, return_tensors="pt").to(device)

                        with torch.no_grad():
                            img_outputs = model.get_image_features(**inputs)
                            img_tensor = img_outputs if isinstance(img_outputs, torch.Tensor) else (img_outputs.image_embeds if hasattr(img_outputs, "image_embeds") else (img_outputs.pooler_output if hasattr(img_outputs, "pooler_output") else img_outputs[0]))
                            img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)
                            vector_list = img_tensor.flatten().cpu().tolist()[:768]

                        insert_data = {
                            "file_name": original_filename,
                            "file_path": public_url,
                            "file_size_kb": file_size_kb,
                            "embedding": vector_list,
                            "tags": uploaded_tags if uploaded_tags else None,
                        }
                        supabase.table("image_embeddings").insert(insert_data).execute()
                        success_count += 1

                    except Exception as e:
                        st.error(f"❌ '{uploaded_file.name}' 처리 중 에러: {e}")

            st.success(f"✅ 총 {success_count}장의 사진이 저장되었습니다!")
            st.session_state.uploader_key = str(uuid.uuid4())
            time.sleep(2)
            st.rerun()


# [탭 3] 관리 기능
with tab_manage:
    st.subheader("🗂️ 이미지 관리")
    top1, top2 = st.columns([1, 1])
    with top1:
        if st.button("🔄 목록 새로고침", use_container_width=True):
            st.rerun()
    with top2:
        cols_n = st.selectbox("한 줄 수", [3, 4, 5, 6, 7], index=2)

    try:
        records = (
            supabase.table("image_embeddings")
            .select("id", "file_name", "file_path", "file_size_kb", "created_at")
            .execute()
            .data
        )

        if not records:
            st.info("현재 저장된 사진이 없습니다.")
        else:
            for start in range(0, len(records), cols_n):
                cols = st.columns(cols_n)
                chunk = records[start:start + cols_n]
                for col, record in zip(cols, chunk):
                    with col:
                        render_manage_card(record)
    except Exception as e:
        st.error(f"에러: {e}")
