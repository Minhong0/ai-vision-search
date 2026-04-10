import streamlit as st
import torch
import os
import uuid
import requests
import time
import datetime  # 날짜 계산을 위해 추가
from PIL import Image
from transformers import AutoProcessor, AutoModel
from supabase import create_client, Client

# ==========================================
# 1. 기본 웹 설정 및 세션 기억력 초기화
# ==========================================
st.set_page_config(page_title="인제 클라우드 갤러리", page_icon="🇰🇷", layout="wide")
st.title("☁️인제 클라우드 갤러리")
st.markdown("사진을 검색 해보세요.")

if "display_count" not in st.session_state:
    st.session_state.display_count = 3
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())

# ==========================================
# 2. AI 모델 및 DB 연결
# ==========================================
@st.cache_resource
def load_system():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    sb = create_client(url, key)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Bingsu/clip-vit-base-patch32-ko"
    
    model = AutoModel.from_pretrained(model_id, use_safetensors=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    return sb, model, processor, device

with st.spinner('AI 엔진을 깨우는 중입니다...'):
    supabase, model, processor, device = load_system()

# ==========================================
# 3. 화면 탭 구성
# ==========================================
tab_search, tab_upload, tab_manage = st.tabs(["🔍 사진 검색", "☁️ 사진 업로드", "🗑️ 갤러리 관리"])

# ------------------------------------------
# [탭 1] 검색 기능 (하이브리드 필터 적용)
# ------------------------------------------
with tab_search:
    st.subheader("머릿속에 있는 사진을 텍스트로 찾아보세요")
    
    # 메인 검색창
    query = st.text_input("검색어 입력 (예: 안전모 쓴 작업자, 영수증, 바다)", key="search_input")
    
    st.markdown("---")
    st.markdown("#### ⚙️ 상세 필터 설정 (선택사항)")
    
    # 토글(Expander)을 없애고 화면에 바로 노출
    col1, col2, col3 = st.columns(3)
    
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
                min_value=0.0, max_value=100.0, value=1.0, step=0.5
            )
            min_size_kb = int(min_size_mb * 1024)
        else:
            min_size_kb = None
            
    with col3:
        st.markdown("**[유사도 설정]**")
        match_threshold = st.slider("유사도 커트라인", 0.0, 0.5, 0.23, 0.01)
        match_count = st.number_input("최대 출력 개수", min_value=1, max_value=50, value=15)
        
    st.markdown("---")
    
    if query:
        if query != st.session_state.last_query:
            st.session_state.display_count = 3
            st.session_state.last_query = query

    with st.spinner('AI가 사용자가 학습시킨 데이터를 바탕으로 교차 검색 중입니다...'):
            try:
                start_date_str = start_date.strftime("%Y-%m-%d") if start_date else None
                end_date_str = end_date.strftime("%Y-%m-%d") if end_date else None

                # 1. 기본 AI 텍스트 벡터 추출 (✨ 에러 수정된 부분)
                inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    text_outputs = model.get_text_features(**inputs)
                    
                    # 원래 사용하시던 안전한 추출 코드로 복구
                    if isinstance(text_outputs, torch.Tensor):
                        text_tensor = text_outputs
                    elif hasattr(text_outputs, 'text_embeds'):
                        text_tensor = text_outputs.text_embeds
                    elif hasattr(text_outputs, 'pooler_output'):
                        text_tensor = text_outputs.pooler_output
                    else:
                        text_tensor = text_outputs[0]
                    
                    text_tensor = text_tensor / text_tensor.norm(p=2, dim=-1, keepdim=True)

                # 👇👇👇 [핵심] 사용자 주도 학습(파인튜닝) 반영 로직 👇👇👇
                # 검색어와 일치하는 태그를 가진 사진들이 DB에 있는지 확인
                tag_response = supabase.table("image_embeddings").select("embedding").ilike("tags", f"%{query}%").execute()
                tag_records = tag_response.data

                if tag_records: # 사용자가 이 단어로 학습시킨 사진이 존재한다면!
                    st.toast("사용자가 가르쳐준 특징을 검색에 반영합니다! 🧠", icon="✨")
                    
                    # 학습된 사진들의 벡터를 모아서 평균을 냄 (사용자 정의 '개념' 생성)
                    learned_vectors = [torch.tensor(record["embedding"]).to(device) for record in tag_records]
                    learned_tensor = torch.stack(learned_vectors).mean(dim=0)
                    learned_tensor = learned_tensor / learned_tensor.norm(p=2, dim=-1, keepdim=True)

                    # 기존 AI 지식(텍스트) 40% + 사용자가 가르쳐준 지식(사진들) 60% 혼합
                    final_tensor = (text_tensor * 0.4) + (learned_tensor * 0.6)
                    final_tensor = final_tensor / final_tensor.norm(p=2, dim=-1, keepdim=True)
                else:
                    # 학습된 데이터가 없으면 기존 방식대로 검색
                    final_tensor = text_tensor
                # 👆👆👆 ------------------------------------------ 👆👆👆

                query_vector = final_tensor.flatten().cpu().tolist()[:512]

                # DB 하이브리드 검색 호출
                response = supabase.rpc("match_images", {
                    "query_embedding": query_vector,
                    "match_threshold": match_threshold,
                    "match_count": match_count,
                    "filter_start_date": start_date_str,
                    "filter_end_date": end_date_str,
                    "filter_min_size_kb": min_size_kb
                }).execute()
                
                results = response.data
                
                # 결과 출력
                if results and len(results) > 0:
                    st.success(f"🎉 필터 조건에 맞는 총 {len(results)}장의 사진을 찾았습니다!")
                    displayed_results = results[:st.session_state.display_count]
                    
                    cols = st.columns(3)
                    for idx, result in enumerate(displayed_results):
                        with cols[idx % 3]:
                            try:
                                st.image(result['file_path'], use_container_width=True)
                                
                                # 화면 표기용 데이터 정제
                                raw_size = result.get('file_size_kb')
                                file_size = int(raw_size) if raw_size is not None else 0
                                created_date = result.get('created_at', '알 수 없음')[:10] if result.get('created_at') else '최근'
                                
                                st.markdown(f"**{result['file_name']}**")
                                st.caption(f"🎯 유사도: {result['similarity']:.3f} | 💾 {file_size}KB | 📅 {created_date}")
                                
                                img_data = requests.get(result['file_path']).content
                                st.download_button(
                                    label="⬇️ 다운로드",
                                    data=img_data,
                                    file_name=result['file_name'],
                                    mime="image/jpeg",
                                    key=f"dl_search_{result['id']}" 
                                )
                            except Exception as e:
                                st.error(f"이미지 로드 실패")
                    
                    if st.session_state.display_count < len(results):
                        st.markdown("---")
                        if st.button("🔽 더 보기 (Load More)", use_container_width=True):
                            st.session_state.display_count += 3
                            st.rerun()
                else:
                    st.warning("⚠️ 필터 조건이나 검색어에 맞는 사진이 없습니다. 필터를 해제하거나 검색어를 바꿔보세요!")
            except Exception as e:
                st.error(f"❌ 검색 중 에러 발생: {e}")

# ------------------------------------------
# [탭 2] 업로드 기능
# ------------------------------------------
with tab_upload:
    st.subheader("새로운 사진들을 클라우드에 한 번에 업로드합니다")
    
    # 먼저 사진 업로드 창이 노출됩니다.
    uploaded_files = st.file_uploader(
        "이미지 파일 선택 (여러 장 드래그 앤 드롭 가능)", 
        type=['png', 'jpg', 'jpeg'], 
        accept_multiple_files=True,
        key=st.session_state.uploader_key
    )
    
    # 사진이 올라가면 그제서야 아래 로직(태그 입력, 사진 미리보기, 업로드 버튼)이 화면에 나타납니다.
    if uploaded_files:
        st.write(f"총 **{len(uploaded_files)}**장의 사진이 선택되었습니다.")
        
        # 👇👇👇 태그 입력란이 파일 업로드 조건문 안으로 들어왔습니다 👇👇👇
        st.markdown("#### 🏷️ 데이터 학습 태그 (선택사항)")
        uploaded_tags = st.text_input(
            "이 사진들의 특징이나 불량 종류를 입력해주세요 (예: 불량 부품, 스크래치, 모터결함)", 
            placeholder="이곳에 태그를 적어두면 추후 AI 파인튜닝 시 정답 데이터로 활용됩니다!"
        )
        st.markdown("---")
        
        cols = st.columns(5)
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 5]:
                st.image(file, use_container_width=True)
        
        if st.button("🚀 일괄 클라우드 업로드 및 AI 분석"):
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
                        file_options={"content-type": uploaded_file.type}
                    )
                    
                    public_url = supabase.storage.from_("images").get_public_url(safe_filename)
                    
                    img = Image.open(uploaded_file).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        img_outputs = model.get_image_features(**inputs)
                        
                        if isinstance(img_outputs, torch.Tensor):
                            img_tensor = img_outputs
                        elif hasattr(img_outputs, 'image_embeds'):
                            img_tensor = img_outputs.image_embeds
                        elif hasattr(img_outputs, 'pooler_output'):
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
                        "tags": uploaded_tags  # 사용자 입력 태그 저장
                    }
                    supabase.table("image_embeddings").insert(insert_data).execute()
                    
                    success_count += 1
                    
                except Exception as e:
                    st.error(f"❌ '{uploaded_file.name}' 처리 중 에러: {e}")
                
                progress_percent = int(((idx + 1) / len(uploaded_files)) * 100)
                my_bar.progress(progress_percent, text=f"진행 중... ({idx+1}/{len(uploaded_files)} 장 완료)")
            
            st.success(f"✅ 총 {success_count}장의 사진이 성공적으로 저장되었습니다!")
            
            st.session_state.uploader_key = str(uuid.uuid4())
            st.rerun()
# ------------------------------------------
# [탭 3] 관리 및 삭제 기능
# ------------------------------------------
with tab_manage:
    st.subheader("🗑️ 클라우드에 저장된 갤러리 관리")
    
    if st.button("🔄 목록 새로고침"):
        st.rerun()

    try:
        records = supabase.table("image_embeddings").select("id", "file_name", "file_path", "file_size_kb", "created_at").execute().data
        
        if not records:
            st.info("현재 클라우드에 저장된 사진이 없습니다.")
        else:
            st.write(f"총 **{len(records)}**장의 사진이 저장되어 있습니다.")
            
            for record in records:
                col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
                
                with col1:
                    st.image(record['file_path'], width=100)
                with col2:
                    raw_size = record.get('file_size_kb')
                    file_size = int(raw_size) if raw_size is not None else 0
                    created_date = record.get('created_at', '알 수 없음')[:10] if record.get('created_at') else '기존 데이터'
                    
                    st.write(f"**{record['file_name']}**")
                    st.caption(f"💾 {file_size}KB | 📅 {created_date}")
                with col3:
                    try:
                        img_data = requests.get(record['file_path']).content
                        st.download_button(
                            label="⬇️ 다운로드",
                            data=img_data,
                            file_name=record['file_name'],
                            mime="image/jpeg",
                            key=f"dl_manage_{record['id']}" 
                        )
                    except:
                        st.write("다운로드 불가")
                with col4:
                    if st.button("❌ 삭제", key=f"del_{record['id']}"):
                        with st.spinner("삭제 중..."):
                            storage_filename = record['file_path'].split('/')[-1]
                            supabase.storage.from_("images").remove([storage_filename])
                            supabase.table("image_embeddings").delete().eq("id", record['id']).execute()
                            
                            st.success("삭제되었습니다!")
                            st.rerun()
    except Exception as e:
        st.error(f"목록을 불러오는 중 에러가 발생했습니다: {e}")
