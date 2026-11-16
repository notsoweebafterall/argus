"""Retrieval tools for Argus ML: build a critic-failure corpus, index it, and retrieve similar failures.

This module provides:
  - build_critic_failure_corpus(): extract rejected critic runs from MLflow into a JSONL corpus
  - build_retrieval_index(): embed the corpus and persist a numpy cosine index
  - retrieve_similar_failures(query, k): retrieve the top-k most similar past failures
  - RETRIEVE_SIMILAR_FAILURES_SPEC: ToolSpec for agent/MCP registration

Run once (and again whenever mlruns/ grows):
  python -m tools.retrieval_tools build-all
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from agents.contracts import ToolSpec
from pipeline.config import settings

# Local index location (gitignored: data/features/, but this is data/retrieval/)
RETRIEVAL_DIR = settings.workspace_root / "data" / "retrieval"
CORPUS_PATH = RETRIEVAL_DIR / "critic_failures.jsonl"
EMBEDDINGS_PATH = RETRIEVAL_DIR / "embeddings.npy"
DOCS_PATH = RETRIEVAL_DIR / "docs.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)


def _render_doc(failed_checks: list[dict[str, Any]], rationale: str = "") -> str:
    """Render a failed_checks list into a single searchable text string."""
    parts: list[str] = []
    for fc in failed_checks:
        parts.append(
            f"check={fc.get('check_name', '')} "
            f"category={fc.get('category', '')} "
            f"reason={fc.get('reason', '')}"
        )
    if rationale:
        parts.append(f"rationale={rationale}")
    return " | ".join(parts)


def build_critic_failure_corpus(out_path: str | Path | None = None) -> int:
    """Extract all critic-rejected runs from MLflow into a JSONL corpus.

    Requires that the critic stage persisted `critic_failed_checks` as a tag
    (see update_critic_verdict_tag in tools/ml_tools.py). If no failed_checks
    tag is present, the run is skipped with a warning.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from tools.ml_tools import _init_mlflow, EXPERIMENT_NAME

    _init_mlflow()
    client = MlflowClient()
    out = Path(out_path) if out_path else CORPUS_PATH

    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for exp in client.search_experiments():
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="tags.critic_verdict = 'reject'",
                max_results=10000,
            )
            for run in runs:
                tags = run.data.tags
                raw = tags.get("critic_failed_checks")
                if not raw:
                    continue
                try:
                    failed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not failed:
                    continue
                doc = {
                    "run_id": run.info.run_id,
                    "experiment": exp.name,
                    "failed_checks": failed,
                    "text": _render_doc(failed, tags.get("critic_rationale", "")),
                }
                f.write(json.dumps(doc) + "\n")
                n += 1

    return n


def build_retrieval_index(corpus_path: str | Path | None = None) -> int:
    """Embed the corpus and persist embeddings.npy + docs.json."""
    from sentence_transformers import SentenceTransformer

    corpus = Path(corpus_path) if corpus_path else CORPUS_PATH
    if not corpus.exists() or corpus.stat().st_size == 0:
        raise SystemExit(
            f"corpus empty or missing at {corpus}. Run build_critic_failure_corpus() first."
        )

    docs = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not docs:
        raise SystemExit("corpus has zero documents")

    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode([d["text"] for d in docs], normalize_embeddings=True).astype("float32")

    np.save(EMBEDDINGS_PATH, vecs)
    DOCS_PATH.write_text(json.dumps(docs), encoding="utf-8")
    return len(docs)


_MODEL = None
_VECS = None
_DOCS = None


def _ensure_loaded() -> None:
    global _MODEL, _VECS, _DOCS
    if _VECS is not None:
        return
    if not EMBEDDINGS_PATH.exists() or not DOCS_PATH.exists():
        raise FileNotFoundError(
            f"retrieval index not found at {RETRIEVAL_DIR}. "
            "Run `python -m tools.retrieval_tools build-all` first."
        )
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer(EMBED_MODEL)
    _VECS = np.load(EMBEDDINGS_PATH)
    _DOCS = json.loads(DOCS_PATH.read_text(encoding="utf-8"))


def retrieve_similar_failures(query: str, k: int = 3) -> dict[str, Any]:
    """Return up to k past critic failures most similar to the query string."""
    _ensure_loaded()
    q = _MODEL.encode([query], normalize_embeddings=True).astype("float32")[0]
    sims = _VECS @ q
    order = np.argsort(-sims)[:k]
    matches = [
        {
            "run_id": _DOCS[i]["run_id"],
            "score": float(sims[i]),
            "failed_checks": _DOCS[i]["failed_checks"],
        }
        for i in order
        if sims[i] > 0.0
    ]
    return {"matches": matches}


RETRIEVE_SIMILAR_FAILURES_SPEC = ToolSpec(
    name="retrieve_similar_failures",
    description=(
        "Retrieve past critic failures most similar to a given failure description. "
        "Returns run_id, similarity score, and the failed_checks that occurred in that run. "
        "Useful for informing whether a recurring failure pattern is being observed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Description of the current failure."},
            "k": {"type": "integer", "description": "Number of matches to return.", "default": 3},
        },
        "required": ["query"],
    },
    handler=retrieve_similar_failures,
)


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "build-all"
    if cmd in ("build-corpus", "build-all"):
        n = build_critic_failure_corpus()
        print(f"corpus: {n} documents -> {CORPUS_PATH}")
    if cmd in ("build-index", "build-all"):
        n = build_retrieval_index()
        print(f"index: {n} documents -> {RETRIEVAL_DIR}")