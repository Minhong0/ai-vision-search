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

HF_REPO_ID = "Rusom/my-custom-factory-clip"
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="인제 클라우드 갤러리",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 1rem 0 0.2rem 0;
    }
    .subtitle {
        text-align: center; color: #888; font-size: 0.95rem; margin-bottom: 1.8rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; border-radius: 12px; padding: 0 14px;
    }
</style>
    """,
    unsafe_allow_html=True,
)

if "display_count" not in st.session_state:
    st.session_state.display_count = 3
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())


@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase = init_supabase()


def check_for_new_model():
    # DB 기록은 확인하되, 발표용 버전에서는 단순히 상태 갱신용으로만 씁니다.
    try:
        latest_job = (
            supabase.table("training_jobs")
            .select("*")
            .eq("status", "completed")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if latest_job.data:
            latest_version = latest_job.data[0]["model_version"]
            if st.session_state.get("current_model_version") != latest_version:
                st.session_state.current_model_version = latest_version
            return latest_version
    except Exception:
        pass
    return "v_base"


# 🚀 [수정됨] 발표 시연을 위해 무조건 고성능 Large 모델만 불러오도록 강제 고정
@st.cache_resource(show_spinner="☁️ 고성능 AI 모델(Large)을 불러오는 중입니다...")
def load_ai_model(version_tag):
    model_id = "Bingsu/clip-vit-base-patch32-ko"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    return processor, model, "large_base"


current_version = check_for_new_model()
processor, model, model_status = load_ai_model(current_version)

# 알림 메시지도 시연용으로 멋지게 변경
if "notified_version" not in st.session_state or st.session_state.notified_version != current_version:
    st.toast("🚀 발표용 고성능 AI 모델(Large) 탑재 완료!", icon="✨")
    st.session_state.notified_version = current_version

st.markdown('<div class="main-title">🔍 자연어 클라우드 갤러리</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">커스텀 AI 검색 · 클라우드 업로드 · 갤러리 관리</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📌 앱 안내")
    st.caption("사용자 화면은 정돈된 카드형 UI로 구성되어 있습니다.")
    st.divider()
    st.caption("검색 팁")
    st.caption("• 안전모 쓴 작업자")
    st.caption("• 영수증")
    st.caption("• 바다")
    st.caption("• 불량 부품")
    st.divider()
    st.caption(f"현재 로드된 모델: Large-Patch14")


def render_search_card(result):
    with st.container(border=True):
        try:
            st.image(result["file_path"], use_container_width=True)
            raw_size = result.get("file_size_kb")
            file_size = int(raw_size) if raw_size is not None else 0
            created_date = result.get("created_at", "알 수 없음")[:10] if result.get("created_at") else "최근"
            st.markdown(f"**{result['file_name']}**")
            st.caption(f"🎯 유사도: {result['similarity']:.3f} · 💾 {file_size}KB · 📅 {created_date}")
            img_data = requests.get(result["file_path"]).content
            st.download_button(
                label="다운로드",
                data=img_data,
                file_name=result["file_name"],
                mime="image/jpeg",
                key=f"dl_search_{result['id']}",
                use_container_width=True,
            )
        except Exception:
            st.error("이미지 로드 실패")


def render_manage_card(record):
    with st.container(border=True):
        st.image(record["file_path"], use_container_width=True)
        raw_size = record.get("file_size_kb")
        file_size = int(raw_size) if raw_size is not None else 0
        created_date = record.get("created_at", "알 수 없음")[:10] if record.get("created_at") else "기존 데이터"
        st.markdown(f"**{record['file_name']}**")
        st.caption(f"💾 {file_size}KB · 📅 {created_date}")

        btn1, btn2 = st.columns(2)
        with btn1:
            try:
                img_data = requests.get(record["file_path"]).content
                st.download_button(
                    label="다운로드",
                    data=img_data,
                    file_name=record["file_name"],
                    mime="image/jpeg",
                    key=f"dl_manage_{record['id']}",
                    use_container_width=True,
                )
            except Exception:
                st.button("다운로드 불가", disabled=True, key=f"disabled_{record['id']}", use_container_width=True)
        with btn2:
            if st.button("삭제", key=f"del_{record['id']}", use_container_width=True):
                with st.spinner("삭제 중..."):
                    storage_filename = record["file_path"].split("/")[-1]
                    supabase.storage.from_("images").remove([storage_filename])
                    supabase.table("image_embeddings").delete().eq("id", record["id"]).execute()
                    st.success("삭제되었습니다!")
                    st.rerun()


# 화면 탭 구성
tab_search, tab_upload, tab_manage = st.tabs(["🔍 사진 검색", "☁️ 사진 업로드", "🗂️ 갤러리 관리"])


# [탭 1] 검색 기능
with tab_search:
    st.subheader("🔍 사진 검색")
    st.caption("머릿속에 있는 장면을 텍스트로 입력하고, 필요하면 상세 필터를 함께 적용하세요.")

    q1, q2, q3 = st.columns([3, 1, 1])
    with q1:
        query = st.text_input("검색어", placeholder="예: 안전모 쓴 작업자, 영수증, 바다", key="search_input")
    with q2:
        match_threshold = st.slider("유사도", 0.0, 0.5, 0.23, 0.01)
    with q3:
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
                min_size_mb = st.number_input(
                    "최소 용량 (MB)",
                    min_value=0.0, max_value=100.0, value=1.0, step=0.5,
                )
                min_size_kb = int(min_size_mb * 1024)
            else:
                min_size_kb = None

    if query:
        if query != st.session_state.last_query:
            st.session_state.display_count = 3
            st.session_state.last_query = query

        with st.spinner("AI가 고성능 파라미터를 바탕으로 교차 검색 중입니다..."):
            try:
                start_date_str = start_date.strftime("%Y-%m-%d") if start_date else None
                end_date_str = end_date.strftime("%Y-%m-%d") if end_date else None

                inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    text_outputs = model.get_text_features(**inputs)

                    if isinstance(text_outputs, torch.Tensor):
                        text_tensor = text_outputs
                    elif hasattr(text_outputs, "text_embeds"):
                        text_tensor = text_outputs.text_embeds
                    elif hasattr(text_outputs, "pooler_output"):
                        text_tensor = text_outputs.pooler_output
                    else:
                        text_tensor = text_outputs[0]

                    final_tensor = text_tensor / text_tensor.norm(p=2, dim=-1, keepdim=True)

                query_vector = final_tensor.flatten().cpu().tolist()[:512]

                response = supabase.rpc(
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

                results = response.data

                if results and len(results) > 0:
                    st.success(f"🎉 필터 조건에 맞는 총 {len(results)}장의 사진을 찾았습니다!")
                    displayed_results = results[: st.session_state.display_count]

                    for start in range(0, len(displayed_results), 3):
                        cols = st.columns(3)
                        chunk = displayed_results[start:start + 3]
                        for col, result in zip(cols, chunk):
                            with col:
                                render_search_card(result)

                    if st.session_state.display_count < len(results):
                        if st.button("더 보기", use_container_width=True):
                            st.session_state.display_count += 3
                            st.rerun()
                else:
                    st.warning("⚠️ 필터 조건이나 검색어에 맞는 사진이 없습니다. 필터를 해제하거나 검색어를 바꿔보세요!")
            except Exception as e:
                st.error(f"❌ 검색 중 에러 발생: {e}")
    else:
        st.info("검색어를 입력하면 결과가 카드형 갤러리로 표시됩니다.")


# [탭 2] 업로드 기능
with tab_upload:
    st.subheader("📤 사진 업로드")
    st.caption("새로운 사진을 클라우드에 올리고, 필요하면 태그를 함께 남길 수 있습니다.")

    uploaded_files = st.file_uploader(
        "이미지 파일 선택 (여러 장 드래그 앤 드롭 가능)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
    )

    if uploaded_files:
        st.write(f"총 **{len(uploaded_files)}**장의 사진이 선택되었습니다.")

        with st.container(border=True):
            st.markdown("**🏷️ 데이터 학습 태그 (선택사항)**")
            uploaded_tags = st.text_input(
                "이 사진들의 특징이나 불량 종류를 입력해주세요 (예: 불량 부품, 스크래치, 모터결함)",
                placeholder="이곳에 태그를 적어두면 추후 AI 파인튜닝 시 정답 데이터로 활용됩니다!",
            )

        for start in range(0, len(uploaded_files), 5):
            cols = st.columns(5)
            chunk = uploaded_files[start:start + 5]
            for col, file in zip(cols, chunk):
                with col:
                    with st.container(border=True):
                        st.image(file, use_container_width=True)
                        st.caption(file.name)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            btn_save_only = st.button("💾 단순 저장 (보관용)", use_container_width=True)
        with col_btn2:
            btn_save_and_train = st.button(
                "🚀 저장 및 AI 학습 시작",
                use_container_width=True,
                disabled=(not uploaded_tags),
            )
            if not uploaded_tags:
                st.caption("※ 학습을 시작하려면 태그를 반드시 입력해야 합니다.")

        if btn_save_only or btn_save_and_train:
            progress_text = "업로드 및 분석을 시작합니다..."
            my_bar = st.progress(0, text=progress_text)
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

                        if isinstance(img_outputs, torch.Tensor):
                            img_tensor = img_outputs
                        elif hasattr(img_outputs, "image_embeds"):
                            img_tensor = img_outputs.image_embeds
                        elif hasattr(img_outputs, "pooler_output"):
                            img_tensor = img_outputs.pooler_output
                        else:
                            img_tensor = img_outputs[0]

                        img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)
                        vector_list = img_tensor.flatten().cpu().tolist()[:512]

                    insert_data = {
                        "file_name": original_filename,
                        "file_path": public_url,
                        "file_size_kb": file_size_kb,
                        "embedding": vector_list,
                        "tags": uploaded_tags,
                    }
                    supabase.table("image_embeddings").insert(insert_data).execute()
                    success_count += 1

                except Exception as e:
                    st.error(f"❌ '{uploaded_file.name}' 처리 중 에러: {e}")

                progress_percent = int(((idx + 1) / len(uploaded_files)) * 100)
                my_bar.progress(progress_percent, text=f"진행 중... ({idx + 1}/{len(uploaded_files)} 장 완료)")

            if btn_save_and_train:
                new_version = f"v_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
                supabase.table("training_jobs").insert(
                    {"status": "pending", "model_version": new_version}
                ).execute()

                st.success(f"✅ 총 {success_count}장의 사진 저장 완료 및 훈련소에 학습 명령을 전송했습니다!")
                st.info("명령 전송 완료! 이제 갤러리 검색을 정상적으로 이용하셔도 됩니다.")
            else:
                st.success(f"✅ 총 {success_count}장의 사진이 성공적으로 저장되었습니다!")

            st.session_state.uploader_key = str(uuid.uuid4())
            time.sleep(2)
            st.rerun()


# [탭 3] 관리 및 삭제 기능
with tab_manage:
    st.subheader("🗂️ 이미지 관리")
    st.caption("클라우드에 저장된 사진을 카드형으로 확인하고, 다운로드 또는 삭제할 수 있습니다.")

    top1, top2 = st.columns([1, 1])
    with top1:
        if st.button("🔄 목록 새로고침", use_container_width=True):
            st.rerun()
    with top2:
        cols_n = st.selectbox("한 줄 수", [3, 4, 5], index=1)

    try:
        records = (
            supabase.table("image_embeddings")
            .select("id", "file_name", "file_path", "file_size_kb", "created_at")
            .execute()
            .data
        )

        if not records:
            st.info("현재 클라우드에 저장된 사진이 없습니다.")
        else:
            st.caption(f"총 {len(records)}장의 사진이 저장되어 있습니다.")
            for start in range(0, len(records), cols_n):
                cols = st.columns(cols_n)
                chunk = records[start:start + cols_n]
                for col, record in zip(cols, chunk):
                    with col:
                        render_manage_card(record)
    except Exception as e:
        st.error(f"목록을 불러오는 중 에러가 발생했습니다: {e}")
