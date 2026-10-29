"""Streamlit interface. Run with `streamlit run -m contract_assistant.app` from src."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from contract_assistant.cli import build_index


@st.cache_resource
def index_for(data_path: str, max_chars: int, overlap: int, retriever: str, k1: float, b: float):
    return build_index(Path(data_path), {"max_chars": max_chars, "overlap": overlap, "retriever": retriever, "k1": k1, "b": b})


st.set_page_config(page_title="Contract Intelligence Assistant", layout="wide")
st.title("Contract Intelligence Assistant")
st.caption("Search CUAD contract passages. Results are evidence excerpts, not legal advice.")
try:
    default_config = json.loads(Path("configs/final.json").read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    default_config = {"max_chars": 1800, "overlap": 180, "abstention_threshold": 0.0}
with st.sidebar:
    data_path = st.text_input("Dataset directory", "data/legalbench-rag")
    top_k = st.slider("Maximum passages", 1, 10, 5)
    threshold = st.number_input("Minimum retrieval score", min_value=0.0, value=float(default_config["abstention_threshold"]), step=0.25)
    max_chars = st.slider("Chunk size (characters)", 200, 6000, int(default_config["max_chars"]), 100)
    overlap = st.slider("Chunk overlap", 0, max(0, int(max_chars) - 1), min(int(default_config["overlap"]), int(max_chars) - 1), 20)
question = st.text_area("Ask about a contract", placeholder="When does the exclusivity period end?")
if st.button("Search", type="primary"):
    if not question.strip():
        st.warning("Enter a question to search.")
    else:
        try:
            with st.spinner("Searching indexed contracts…"):
                index = index_for(data_path, int(max_chars), int(overlap), default_config.get("retriever", "bm25"),
                                  float(default_config.get("k1", 1.5)), float(default_config.get("b", 0.75)))
                results = index.search(question, top_k, threshold)
            if not results:
                st.info("No relevant passage found. Try rephrasing or check another contract.")
            for hit in results:
                with st.expander(f"{hit.rank}. {hit.document_id} · {hit.score:.3f}", expanded=hit.rank == 1):
                    st.caption(f"Original characters {hit.start}–{hit.end} · BM25 {hit.score:.4f}")
                    st.text(hit.text)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            st.error(f"Could not load the dataset: {exc}")
