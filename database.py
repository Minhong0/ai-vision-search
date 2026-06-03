import streamlit as st
from supabase import create_client


@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase = init_supabase()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_gallery() -> list:
    """갤러리 전체 목록 (30초 캐시)"""
    return (
        supabase.table("image_embeddings")
        .select("id, file_name, file_path, file_size_kb, created_at, tags")
        .order("created_at", desc=True)
        .execute()
        .data
    ) or []


@st.cache_data(ttl=60, show_spinner=False)
def fetch_stats() -> list:
    """통계용 데이터 (60초 캐시)"""
    return (
        supabase.table("image_embeddings")
        .select("file_size_kb, tags, created_at")
        .execute()
        .data
    ) or []
