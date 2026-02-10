import glob
import os
from typing import Dict, List, Tuple

import pandas as pd
import requests
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_KNOWLEDGE_TEXT = """\
Rare-Disease Structural Anomaly Detector — Knowledge Base

This app visualizes protein structural risk signals from AlphaMissense and AlphaFold.

Key concepts:
- Anomaly Score: AlphaMissense pathogenicity score per variant (0 to 1). Higher implies higher predicted risk.
- Instability Plot (2D): per-residue average anomaly score along the sequence.
- Model Confidence (pLDDT): per-residue confidence from AlphaFold PDB B-factors (0 to 100).
- Hotspots: clusters of high-risk residues that are close in 3D space (DBSCAN on C-alpha coordinates).

How to interpret:
- Tall peaks in the Instability Plot show residues with high predicted impact.
- pLDDT indicates confidence; low pLDDT regions are less reliable structurally.
- Hotspots indicate spatial grouping of risky residues, not just sequence proximity.

Limitations:
- These are research predictions, not clinical diagnostics.
- Visual overlays may be used to emphasize regions of interest.
"""


def ensure_knowledge_base(knowledge_dir: str = "knowledge") -> str:
    os.makedirs(knowledge_dir, exist_ok=True)
    overview_path = os.path.join(knowledge_dir, "overview.txt")
    if not os.path.exists(overview_path):
        with open(overview_path, "w", encoding="utf-8") as handle:
            handle.write(DEFAULT_KNOWLEDGE_TEXT)

    pdf_path = os.path.join(knowledge_dir, "overview.pdf")
    if not os.path.exists(pdf_path):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in DEFAULT_KNOWLEDGE_TEXT.splitlines():
                pdf.cell(0, 8, line, ln=True)
            pdf.output(pdf_path)
        except Exception:
            # If PDF generation fails, keep the text file only
            pass

    return overview_path


def _build_cache_summary(cache_dir: str = "data/cache") -> str:
    if not os.path.isdir(cache_dir):
        return ""

    rows = []
    for path in glob.glob(os.path.join(cache_dir, "*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if df.empty:
            continue

        uniprot_id = os.path.splitext(os.path.basename(path))[0]
        residue_count = int(df["residue_num"].nunique()) if "residue_num" in df else 0
        mean_score = float(df["am_pathogenicity"].mean()) if "am_pathogenicity" in df else 0.0
        rows.append(
            f"{uniprot_id}: residues={residue_count}, mean_anomaly={mean_score:.3f}"
        )

    if not rows:
        return ""

    header = "Cached protein anomaly summaries:"
    return "\n".join([header] + rows)


def build_corpus(knowledge_dir: str = "knowledge") -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    for path in glob.glob(os.path.join(knowledge_dir, "*.txt")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read().strip()
            if text:
                docs.append({"id": os.path.basename(path), "text": text})
        except Exception:
            continue

    cache_summary = _build_cache_summary()
    if cache_summary:
        docs.append({"id": "cache_summary", "text": cache_summary})

    return docs


def get_knowledge_mtime_key(
    knowledge_dir: str = "knowledge", cache_dir: str = "data/cache"
) -> str:
    paths = glob.glob(os.path.join(knowledge_dir, "*.txt")) + glob.glob(
        os.path.join(cache_dir, "*.csv")
    )
    mtimes = []
    for path in paths:
        try:
            mtimes.append(str(os.path.getmtime(path)))
        except Exception:
            continue
    return "|".join(mtimes)


def build_knowledge_index(
    knowledge_dir: str = "knowledge",
) -> Tuple[TfidfVectorizer, object, List[Dict[str, str]]]:
    docs = build_corpus(knowledge_dir)
    if not docs:
        docs = [{"id": "empty", "text": "No knowledge documents available."}]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform([doc["text"] for doc in docs])
    return vectorizer, matrix, docs


def retrieve_docs(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix,
    docs: List[Dict[str, str]],
    top_k: int = 3,
) -> List[Dict[str, str]]:
    if not query.strip():
        return []
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in ranked[:top_k] if score > 0]


def call_ollama(prompt: str, model: str = "llama3.1") -> str:
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as exc:
        return f"Error calling Ollama: {exc}"


def build_prompt(
    question: str, context_chunks: List[Dict[str, str]], dynamic_context: str
) -> str:
    context_text = "\n\n".join(
        f"[{c['id']}]\n{c['text']}" for c in context_chunks
    )
    return (
        "You are a protein-structure assistant for a research app. "
        "Answer based on the provided context and the current run summary. "
        "If the context is missing, say so briefly.\n\n"
        f"Current run context:\n{dynamic_context}\n\n"
        f"Knowledge context:\n{context_text}\n\n"
        f"Question: {question}\nAnswer:"
    )
