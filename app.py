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


# 🚀 [기능 1] 라디오 버튼 선택에 따라 모델을 동적으로 캐싱 및 로드하는 함수
# 스위치를 누를 때마다 이 함수가 실행되며 자동으로 로딩 스피너가 표시됩니다.
@st.cache_resource(show_spinner="☁️ 선택한 AI 모델(768차원)의 가중치를 메모리에 로드 중입니다...")
def load_ai_model(use_custom):
    if use_custom:
        try:
            processor = AutoProcessor.from_pretrained(HF_REPO_ID)
            model = AutoModel.from_pretrained(HF_REPO_ID).to(device)
            return processor, model, "Custom_Trained_768 (맞춤형)"
        except Exception as e:
            st.error("아직 커스텀 모델이 허깅페이스에 업로드되지 않았습니다. 파인튜닝을 먼저 진행해주세요!")
            # 오류 발생 시 안전하게 기본 모델로 폴백
            processor = AutoProcessor.from_pretrained("Bingsu/clip-vit-large-patch14-ko")
            model = AutoModel.from_pretrained("Bingsu/clip-vit-large-patch14-ko").to(device)
            return processor, model, "Base_Large_768 (기본형)"
    else:
        processor = AutoProcessor.from_pretrained("Bingsu/clip-vit-large-patch14-ko")
        model = AutoModel.from_pretrained("Bingsu/clip-vit-large-patch14-ko").to(device)
        return processor, model, "Base_Large_768 (기본형)"


st.markdown('<div class="main-title">🔍 자연어 클라우드 갤러리</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">커스텀 AI 검색 · 클라우드 업로드 · 갤러리 관리</div>', unsafe_allow_html=True)

# =====================================================================
# 🎛️ 사이드바 UI: 모델 스위칭 컨트롤
# =====================================================================
with st.sidebar:
    st.header("📌 AI 뇌(Model) 스위칭")
    st.caption("발표 시연용: 버튼을 눌러 AI의 성능 차이를 비교하세요.")
    
    # 모델 선택 라디오 버튼
    model_choice = st.radio(
        "테스트할 AI 모델 선택:",
        ["1. 오리지널 범용 모델 (학습 전)", "2. 커스텀 맞춤형 모델 (학습 후)"]
    )
    is_custom = "커스텀" in model_choice
    
    st.divider()
    st.caption("검색 팁")
    st.caption("• 특수 객체 이름")
    st.caption("• 스크래치 난 부품")
    st.caption("• 도로 위 하얀 자동차")

# 선택된 값에 따라 모델 로드 (캐싱되어 있어 전환 속도가 빠릅니다)
processor, model, model_status = load_ai_model(is_custom)

with st.sidebar:
    st.divider()
    st.info(f"현재 가동 중: {model_status}")


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
    st.caption("좌측 사이드바에서 AI 모델을 바꾼 뒤 동일한 검색어로 성능 차이를 확인해보세요.")

    q1, q2, q3 = st.columns([3, 1, 1])
    with q1:
        query = st.text_input("검색어", placeholder="예: 안전모 쓴 작업자, 영수증, 바다", key="search_input")
    with q2:
        match_threshold = st.slider("유사도 커트라인", 0.0, 0.4, 0.18, 0.01)
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

        with st.spinner("AI가 768차원 고성능 분석 중..."):
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

                query_vector = final_tensor.flatten().cpu().tolist()[:768]

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
                    st.warning("⚠️ 사진을 찾지 못했습니다. 커트라인 수치를 낮춰보세요!")
            except Exception as e:
                st.error(f"❌ 검색 중 에러 발생: {e}")
    else:
        st.info("검색어를 입력하면 결과가 카드형 갤러리로 표시됩니다.")


# [탭 2] 업로드 기능
with tab_upload:
    st.subheader("📤 사진 업로드 및 MLOps")
    st.caption("새로운 사진을 올리고 AI를 직접 가르쳐보세요. 훈련 진행 상황을 실시간으로 추적합니다.")

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
                "이 사진들의 특징이나 이름을 입력해주세요",
                placeholder="예: 인제대 마스코트, 불량 부품 A...",
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

        if btn_save_only or btn_save_and_train:
            # 1. 파일 및 벡터 저장 스피너
            with st.spinner("이미지 업로드 및 768차원 임베딩 분석 중..."):
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
                            vector_list = img_tensor.flatten().cpu().tolist()[:768]

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

            if btn_save_only:
                st.success(f"✅ 총 {success_count}장의 사진이 보관용으로 저장되었습니다!")
                st.session_state.uploader_key = str(uuid.uuid4())
                time.sleep(2)
                st.rerun()

            # 🚀 [기능 2] 훈련 과정을 실시간으로 추적하는 동적 상태 표시바 (Polling)
            if btn_save_and_train:
                new_version = f"v_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
                supabase.table("training_jobs").insert(
                    {"status": "pending", "model_version": new_version}
                ).execute()

                # st.status를 활용하여 훈련 상태를 실시간으로 UI에 업데이트
                with st.status("🚀 MLOps 파인튜닝 파이프라인 가동 중...", expanded=True) as status:
                    st.write("1. 📥 클라우드 데이터베이스에 학습 명령 전송 완료")
                    st.write("2. ⏳ 로컬 GPU 서버(train.py)의 작업 시작을 대기 중입니다...")
                    
                    # 훈련소의 상태 변경을 감지하는 무한 루프 (폴링)
                    while True:
                        time.sleep(3) # DB 과부하를 막기 위해 3초 대기
                        check_res = supabase.table("training_jobs").select("status").eq("model_version", new_version).execute()
                        
                        if check_res.data:
                            current_status = check_res.data[0]['status']
                            
                            if current_status == "training":
                                st.write("3. 🧠 로컬 GPU에서 역전파(Backpropagation) 및 파인튜닝 진행 중...")
                            elif current_status == "completed":
                                status.update(label="✅ 허깅페이스 클라우드 자동 배포 완료!", state="complete", expanded=False)
                                break
                            elif current_status == "failed":
                                status.update(label="❌ 파인튜닝 실패 (로컬 터미널 로그를 확인하세요)", state="error", expanded=False)
                                break

                # 학습 완료 메시지 및 안내
                st.success("🎉 파인튜닝이 모두 완료되었습니다!")
                st.info("💡 좌측 사이드바에서 '커스텀 맞춤형 모델' 라디오 버튼을 선택하여 똑똑해진 결과를 확인하세요.")
                st.session_state.uploader_key = str(uuid.uuid4())
                st.cache_resource.clear() # 다음 로드를 위해 캐시 초기화


# [탭 3] 관리 기능
with tab_manage:
    st.subheader("🗂️ 이미지 관리")
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
