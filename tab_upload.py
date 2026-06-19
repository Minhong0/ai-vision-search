import io
import os
import uuid
import time
import numpy as np
import streamlit as st
from PIL import Image
from database import supabase, fetch_gallery, fetch_stats
from embeddings import get_image_embedding
from models import ocr_reader


def _save_image(img: Image.Image, original_filename: str, tags: str | None, use_ocr: bool) -> None:
    ext = os.path.splitext(original_filename)[1] or ".jpg"
    safe_filename = f"{uuid.uuid4().hex}{ext}"

    buf = io.BytesIO()
    img.save(buf, format="JPEG" if ext.lower() in (".jpg", ".jpeg") else "PNG")
    file_bytes = buf.getvalue()
    file_size_kb = len(file_bytes) // 1024

    supabase.storage.from_("images").upload(
        path=safe_filename,
        file=file_bytes,
        file_options={"content-type": "image/jpeg"},
    )
    public_url = supabase.storage.from_("images").get_public_url(safe_filename)

    vector_list = get_image_embedding(img)

    ocr_tag_str = ""
    if use_ocr and ocr_reader is not None:
        ocr_texts = ocr_reader.readtext(np.array(img), detail=0)
        ocr_tag_str = " ".join(ocr_texts).strip()
        if ocr_tag_str:
            st.caption(f"OCR 감지: {ocr_tag_str}")

    merged_tags = " ".join(filter(None, [tags, ocr_tag_str])) or None

    supabase.table("image_embeddings").insert({
        "file_name": original_filename,
        "file_path": public_url,
        "file_size_kb": file_size_kb,
        "embedding": vector_list,
        "tags": merged_tags,
    }).execute()


def _tag_options(key_prefix: str) -> tuple[str, bool]:
    with st.container(border=True):
        st.markdown("**🏷️ 태그 (선택사항)**")
        tags = st.text_input(
            "특징이나 이름을 입력해주세요 (검색 시 태그 일치 점수로 활용됩니다)",
            placeholder="예: 불량 부품, 안전모, 작업자...",
            key=f"{key_prefix}_tags",
        )
        use_ocr = st.checkbox(
            "🔍 OCR 자동 태그 추출 사용",
            value=False,
            disabled=(ocr_reader is None),
            help="easyocr 미설치 시 비활성화됩니다." if ocr_reader is None else "",
            key=f"{key_prefix}_ocr",
        )
    return tags, use_ocr


def _flush_cache() -> None:
    fetch_gallery.clear()
    fetch_stats.clear()


def render():
    tab_file, tab_camera = st.tabs(["📁 파일 업로드", "📸 카메라 촬영"])

    # ── 파일 업로드 탭 ────────────────────────────────────────────────
    with tab_file:
        uploaded_files = st.file_uploader(
            "이미지 파일 선택 (여러 장 드래그 앤 드롭 가능)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=st.session_state.uploader_key,
        )

        if uploaded_files:
            st.write(f"총 **{len(uploaded_files)}**장의 사진이 선택되었습니다.")

            tags, use_ocr = _tag_options("file")

            for s in range(0, len(uploaded_files), 5):
                cols = st.columns(5)
                for col, file in zip(cols, uploaded_files[s:s + 5]):
                    with col:
                        with st.container(border=True):
                            st.image(file, use_container_width=True)
                            st.caption(file.name)

            if st.button("💾 저장", use_container_width=True, key="file_save"):
                with st.spinner("이미지 업로드 및 분석 중..."):
                    success_count = 0
                    for uploaded_file in uploaded_files:
                        try:
                            img = Image.open(uploaded_file).convert("RGB")
                            _save_image(img, uploaded_file.name, tags, use_ocr)
                            success_count += 1
                        except Exception as e:
                            st.error(f"❌ '{uploaded_file.name}' 처리 중 에러: {e}")

                st.success(f"✅ 총 {success_count}장의 사진이 저장되었습니다!")
                _flush_cache()
                st.session_state.uploader_key = str(uuid.uuid4())
                time.sleep(2)
                st.rerun()

    # ── 카메라 촬영 탭 ────────────────────────────────────────────────
    with tab_camera:
        st.caption("모바일 브라우저에서 접속하면 폰 카메라로 바로 촬영할 수 있습니다.")

        photo = st.camera_input("촬영하기", key="camera_input")

        if photo:
            img = Image.open(photo).convert("RGB")
            st.image(img, caption="촬영된 사진", use_container_width=True)

            tags, use_ocr = _tag_options("camera")

            custom_name = st.text_input(
                "파일명 (선택사항)",
                placeholder="예: 안전모_작업자.jpg",
                key="camera_filename",
            )

            if st.button("💾 저장", use_container_width=True, key="camera_save"):
                filename = custom_name.strip() or f"camera_{uuid.uuid4().hex[:8]}.jpg"
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    filename += ".jpg"

                with st.spinner("사진 업로드 및 분석 중..."):
                    try:
                        _save_image(img, filename, tags, use_ocr)
                        st.success(f"✅ '{filename}' 저장 완료!")
                        _flush_cache()
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 저장 중 에러: {e}")
