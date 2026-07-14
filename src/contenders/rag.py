"""Step 3 — RAG: retrieve, then answer.

The corpus is deliberately adversarial: the 126 Phase A hospital sections (the only
ground truth) are indexed alongside chunks from 15 real CDC/ECDC medical PDFs, which are
about medicine but not about this hospital. A hospital-only index could never surface a
retriever that cannot tell them apart. Every answer records which chunks came back and
whether they were distractors, which is what lets Phase C separate "the retriever missed"
from "the model ignored good context".

One fixed config for all 100 questions — frozen to disk, since tuning k or the template
per question would make the comparison meaningless.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .answers import load_test
from .config import INDEX_DIR, RAW_MEDICAL, RESULTS_DIR, ROOT, SECTIONS_PATH
from .models import EMBED_MODEL, PRODUCERS, RAG
from .runner import run_producer

TOP_K = 3
DISTRACTOR_CHUNK_CHARS = 1200   # ~300 tok, comparable to a mean hospital section
EMBED_BATCH = 32

INDEX_PATH = INDEX_DIR / "corpus.faiss"
META_PATH = INDEX_DIR / "corpus.jsonl"
RAG_CONFIG_PATH = RESULTS_DIR / "rag_config.json"

PROMPT_TEMPLATE = """\
Use ONLY the hospital document excerpts below to answer the question. If the \
excerpts do not contain the answer, say so — do not guess.

<documents>
{context}
</documents>

Question: {question}"""


def _section_text(sec: dict) -> str:
    """Flatten a Phase A section into one embeddable string, the same way generate.py
    flattens tables (cells joined with ' | ')."""
    parts = [sec["heading"]]
    for b in sec["content"]:
        if b["kind"] == "para":
            parts.append(b["text"])
        elif b["kind"] == "bullets":
            parts += [f"- {i}" for i in b["items"]]
        elif b["kind"] == "table":
            parts.append(" | ".join(b["header"]))
            parts += [" | ".join(r) for r in b["rows"]]
    return "\n".join(parts)


def _hospital_chunks() -> list[dict]:
    """The 126 verified sections — the ONLY source of ground truth."""
    if not SECTIONS_PATH.exists():
        sys.exit(f"{SECTIONS_PATH.relative_to(ROOT)} is missing — run "
                 f"`python -m build_dataset segment` first.")
    out = []
    for line in SECTIONS_PATH.open(encoding="utf-8"):
        sec = json.loads(line)
        out.append({"doc_id": sec["doc_id"], "section": sec["heading"],
                    "source_doc": sec["source_doc"], "text": _section_text(sec),
                    "is_distractor": False})
    return out


def _distractor_chunks() -> list[dict]:
    """Real CDC/ECDC reports — medical, but nothing to do with Serendib. Their job is to
    make retrieval able to fail, so that succeeding means something."""
    if not RAW_MEDICAL.exists() or not any(RAW_MEDICAL.glob("*.pdf")):
        print(f"  ! no PDFs in {RAW_MEDICAL.name}/ — indexing hospital docs only.\n"
              f"    Retrieval will look better than it deserves to; see docs/phase-b.md.")
        return []
    import fitz
    out = []
    for pdf in sorted(RAW_MEDICAL.glob("*.pdf")):
        text = "\n".join(page.get_text() for page in fitz.open(pdf))
        for i in range(0, len(text), DISTRACTOR_CHUNK_CHARS):
            chunk = text[i:i + DISTRACTOR_CHUNK_CHARS].strip()
            if len(chunk) > 200:
                out.append({"doc_id": f"DISTRACTOR:{pdf.stem[:40]}",
                            "section": f"chunk {i // DISTRACTOR_CHUNK_CHARS}",
                            "source_doc": pdf.name, "text": chunk, "is_distractor": True})
    return out


def _embed(texts: list[str]) -> np.ndarray:
    """nomic-embed-text via Ollama. L2-normalised, so a FAISS inner-product index is
    exactly cosine similarity."""
    import ollama
    vecs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        r = ollama.embed(model=EMBED_MODEL, input=batch)
        vecs.extend(r["embeddings"])
        print(f"  embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}", end="\r")
    arr = np.asarray(vecs, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def build_index() -> None:
    import faiss
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    hospital = _hospital_chunks()
    distractors = _distractor_chunks()
    corpus = hospital + distractors
    print(f"Indexing {len(corpus)} chunks: {len(hospital)} hospital sections "
          f"+ {len(distractors)} distractor chunks")

    vecs = _embed([c["text"] for c in corpus])
    index = faiss.IndexFlatIP(vecs.shape[1])   # exact cosine — the corpus is small
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))

    with META_PATH.open("w", encoding="utf-8") as fh:
        for c in corpus:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAG_CONFIG_PATH.write_text(json.dumps({
        "embed_model": EMBED_MODEL, "dim": int(vecs.shape[1]), "top_k": TOP_K,
        "index": "faiss.IndexFlatIP (L2-normalised = exact cosine)",
        "chunks_total": len(corpus), "chunks_hospital": len(hospital),
        "chunks_distractor": len(distractors),
        "distractor_chunk_chars": DISTRACTOR_CHUNK_CHARS,
        "prompt_template": PROMPT_TEMPLATE,
    }, indent=2), encoding="utf-8")

    print(f"\n  ✓ {INDEX_PATH.name} ({vecs.shape[0]} x {vecs.shape[1]})  ->  "
          f"{INDEX_DIR.name}/")
    print(f"  ✓ frozen config -> {RAG_CONFIG_PATH.name}")


class Retriever:
    def __init__(self) -> None:
        import faiss
        if not INDEX_PATH.exists():
            sys.exit("No index — run `python -m contenders rag --build-index` first "
                     "(free and fully local).")
        self.index = faiss.read_index(str(INDEX_PATH))
        self.meta = [json.loads(l) for l in META_PATH.open(encoding="utf-8")]

    def search(self, question: str, k: int = TOP_K) -> list[dict]:
        vec = _embed([question])
        scores, idx = self.index.search(vec, k)
        hits = []
        for score, i in zip(scores[0], idx[0]):
            c = self.meta[int(i)]
            hits.append({"doc_id": c["doc_id"], "section": c["section"],
                         "score": round(float(score), 4),
                         "is_distractor": c["is_distractor"], "text": c["text"]})
        return hits


def cmd_rag(args) -> None:
    if args.build_index:
        build_index()
        return

    questions = load_test()
    if args.limit:
        questions = questions[:args.limit]

    producers = [p for p in PRODUCERS if p.method == RAG]
    if args.local_only:
        producers = [p for p in producers if p.is_local]

    retriever = Retriever()
    print(f"Step 3 · RAG — retrieve top-{TOP_K}, then answer  (one fixed config)")
    print(f"{len(producers)} producers x {len(questions)} questions")

    def build_prompt(q: dict):
        hits = retriever.search(q["question"])
        context = "\n\n".join(f"[{h['doc_id']} · {h['section']}]\n{h['text']}" for h in hits)
        user = PROMPT_TEMPLATE.format(context=context, question=q["question"])
        # Log the provenance, not a second copy of the corpus.
        return user, [{k: v for k, v in h.items() if k != "text"} for h in hits]

    for p in producers:
        run_producer(p, questions, build_prompt=build_prompt)
