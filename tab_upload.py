import os
import uuid
import time
import numpy as np
import streamlit as st
from PIL import Image
from database import supabase, fetch_gallery, fetch_stats
from embeddings import get_image_embedding
from models import ocr_reader


def render():
    st.subheader("📤 사진 업로드")

    uploaded_files = st.file_uploader(
        "이미지 파일 선택 (여러 장 드래그 앤 드롭 가능)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
    )

    if not uploaded_files:
        return

    st.write(f"총 **{len(uploaded_files)}**장의 사진이 선택되었습니다.")

    with st.container(border=True):
        st.markdown("**🏷️ 태그 (선택사항)**")
        uploaded_tags = st.text_input(
            "이 사진들의 특징이나 이름을 입력해주세요 (검색 시 태그 일치 점수로 활용됩니다)",
            placeholder="예: 불량 부품, 안전모, 작업자...",
        )
        use_ocr = st.checkbox(
            "🔍 OCR 자동 태그 추출 사용 (이미지 안의 텍스트를 읽어 태그에 추가)",
            value=False,
            disabled=(ocr_reader is None),
            help="easyocr 미설치 시 비활성화됩니다." if ocr_reader is None else "",
        )

    for s in range(0, len(uploaded_files), 5):
        cols = st.columns(5)
        for col, file in zip(cols, uploaded_files[s:s + 5]):
            with col:
                with st.container(border=True):
                    st.image(file, use_container_width=True)
                    st.caption(file.name)

    if not st.button("💾 저장", use_container_width=True):
        return

    with st.spinner("이미지 업로드 및 분석 중..."):
        success_count = 0
        for uploaded_file in uploaded_files:
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
                vector_list = get_image_embedding(img)

                ocr_tag_str = ""
                if use_ocr and ocr_reader is not None:
                    ocr_texts = ocr_reader.readtext(np.array(img), detail=0)
                    ocr_tag_str = " ".join(ocr_texts).strip()
                    if ocr_tag_str:
                        st.caption(f"OCR 감지 ({original_filename}): {ocr_tag_str}")

                merged_tags = " ".join(filter(None, [uploaded_tags, ocr_tag_str])) or None

                supabase.table("image_embeddings").insert({
                    "file_name": original_filename,
                    "file_path": public_url,
                    "file_size_kb": file_size_kb,
                    "embedding": vector_list,
                    "tags": merged_tags,
                }).execute()
                success_count += 1

            except Exception as e:
                st.error(f"❌ '{uploaded_file.name}' 처리 중 에러: {e}")

    st.success(f"✅ 총 {success_count}장의 사진이 저장되었습니다!")
    fetch_gallery.clear()
    fetch_stats.clear()
    st.session_state.uploader_key = str(uuid.uuid4())
    time.sleep(2)
    st.rerun()
