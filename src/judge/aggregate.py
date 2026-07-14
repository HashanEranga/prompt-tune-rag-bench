"""Step 5 — the master results table, and the three cuts the write-up is built from.

Joins scores.csv to answers.jsonl on (question_id, producer), so quality sits in the same
row as the cost and latency it was bought with. Rule #4: there is no single winner, and a
table without cost and speed columns cannot show that.
"""
from __future__ import annotations

import json

import pandas as pd

from contenders.answers import load_all as load_answers, load_test
from contenders.models import AB_PAIRS, BY_KEY, PRODUCERS, TRIAD, load_jobs

from .config import MASTER_PATH
from .rubric import DIMENSIONS, WEIGHTS
from .scores import load_all as load_scores

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = load_scores()
    if not scores:
        raise SystemExit("results/scores.csv is empty — run `judge submit` then "
                         "`judge collect` first.")
    s = pd.DataFrame(scores)
    for d in DIMENSIONS:
        s[d] = s[d].astype(int)
    s["weighted"] = s["weighted"].astype(float)
    s["safety_flag"] = s["safety_flag"].astype(str).str.lower() == "true"
    s["judge_cost_usd"] = s["cost_usd"].astype(float)

    a = pd.DataFrame(load_answers())
    a = a[a["error"].isna()] if "error" in a else a
    a = a[["question_id", "producer", "latency_s", "cost_usd", "retrieved"]]
    a = a.rename(columns={"cost_usd": "answer_cost_usd"})

    df = s.merge(a, on=["question_id", "producer"], how="left", validate="one_to_one")
    return df, s


def build_master(df: pd.DataFrame) -> pd.DataFrame:
    jobs = load_jobs()
    rows = []
    for p in PRODUCERS:
        g = df[df["producer"] == p.key]
        if g.empty:
            continue
        train = jobs.get(p.key, {}).get("train_cost_usd", 0.0)
        answer_cost = float(g["answer_cost_usd"].sum())
        rows.append({
            "producer": p.key,
            "method": p.method,
            "model": p.model,
            "size": p.size or "",
            "n": len(g),
            **{d: round(float(g[d].mean()), 3) for d in DIMENSIONS},
            "weighted": round(float(g["weighted"].mean()), 3),
            "safety_flag_rate": round(float(g["safety_flag"].mean()), 3),
            "mean_latency_s": round(float(g["latency_s"].mean()), 2),
            "answer_cost_usd": round(answer_cost, 4),
            "train_cost_usd": round(train, 2),
            "total_cost_usd": round(answer_cost + train, 4),
            "judge_cost_usd": round(float(g["judge_cost_usd"].sum()), 4),
        })
    m = pd.DataFrame(rows).sort_values("weighted", ascending=False).reset_index(drop=True)
    return m


def _retrieval_cut(df: pd.DataFrame) -> pd.DataFrame | None:
    """The cut only this pipeline can make.

    Every RAG answer logged WHICH chunks it retrieved. Crossing that against faithfulness
    separates failures that look identical in a score table and call for opposite fixes.

    Measured over the real retrieval logs, the top-3 contains:

        the gold section          32/100   — the exact section the Q&A pair came from
        the right doc, wrong sec  38/100   — same document, a neighbouring section
        neither                   30/100   — the retriever went somewhere else entirely

    That middle tier is why this is three-way and not a boolean. Because the judge reads the
    FULL source document (Rule #2), an answer assembled from a neighbouring section of the
    right document can still be perfectly faithful — a boolean hit/miss would score it as a
    retrieval failure and hide that. It is also the tier that says whether chunking, rather
    than embedding, is what needs fixing.
    """
    rag = df[df["method"] == "rag"].copy()
    if rag.empty:
        return None
    gold = {q["id"]: (q["doc_id"], q["section"]) for q in load_test()}

    def tier(row) -> str:
        chunks = row["retrieved"]
        if isinstance(chunks, str):
            chunks = json.loads(chunks)
        if not isinstance(chunks, list):
            return "3. neither"
        doc, section = gold[row["question_id"]]
        if any(c.get("doc_id") == doc and c.get("section") == section for c in chunks):
            return "1. gold section"
        if any(c.get("doc_id") == doc for c in chunks):
            return "2. right doc, wrong section"
        return "3. neither"

    rag["retrieval"] = rag.apply(tier, axis=1)
    out = (rag.groupby(["producer", "retrieval"])
              .agg(n=("weighted", "size"),
                   faithfulness=("faithfulness", "mean"),
                   accuracy=("accuracy", "mean"),
                   weighted=("weighted", "mean"))
              .round(3).reset_index())
    return out


def _show(title: str, frame: pd.DataFrame, cols: list[str] | None = None) -> None:
    print(f"\n  {title}")
    print("  " + "-" * len(title))
    view = frame[cols] if cols else frame
    print("  " + view.to_string(index=False).replace("\n", "\n  "))


def cmd_aggregate(_args) -> None:
    df, _ = _frames()
    master = build_master(df)

    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_PATH, index=False)

    w = ", ".join(f"{d} {v:.0%}" for d, v in WEIGHTS.items())
    print(f"\n  MASTER RESULTS TABLE     ({len(df)} scored answers)")
    print(f"  weighted score = {w}\n")
    print("  " + master.to_string(index=False).replace("\n", "\n  "))

    # 1. The TRIAD — one model, three methods, one judge. Any gap IS the method.
    triad = master[master["producer"].isin(TRIAD)].copy()
    if not triad.empty:
        triad["leg"] = triad["producer"].map(
            {TRIAD[0]: "1. prompted", TRIAD[1]: "2. fine-tuned", TRIAD[2]: "3. + documents"})
        _show("THE TRIAD — llama3.1:8b, three ways (a score gap here can only be the method)",
              triad.sort_values("leg"),
              ["leg", "producer", *DIMENSIONS, "weighted", "mean_latency_s", "total_cost_usd"])

    # 2. The A/B pairs — does retrieval help a small model more than a big one?
    print("\n  RETRIEVAL LIFT — same model, documents vs no documents")
    print("  " + "-" * 52)
    idx = master.set_index("producer")
    for base, rag in AB_PAIRS:
        if base not in idx.index or rag not in idx.index:
            continue
        lift = idx.loc[rag, "weighted"] - idx.loc[base, "weighted"]
        fl = idx.loc[rag, "faithfulness"] - idx.loc[base, "faithfulness"]
        print(f"  {base:<20} {idx.loc[base, 'weighted']:.3f}  ->  {rag:<18} "
              f"{idx.loc[rag, 'weighted']:.3f}   lift {lift:+.3f}  (faithfulness {fl:+.3f})")

    # 3. The ladder — does quality actually keep climbing with size?
    ladder = master[master["size"].isin(["small", "medium", "large"])]
    if not ladder.empty:
        order = {"small": 0, "medium": 1, "large": 2}
        _show("THE FINE-TUNING LADDER — does quality keep climbing?",
              ladder.assign(_o=ladder["size"].map(order)).sort_values("_o"),
              ["size", "producer", "model", *DIMENSIONS, "weighted", "mean_latency_s"])

    # 4. Retriever missed, or model ignored good context?
    cut = _retrieval_cut(df)
    if cut is not None and not cut.empty:
        _show("RAG DIAGNOSIS — what did the retriever actually put in front of the model?", cut)
        print("\n    1. gold section, but low faithfulness -> the MODEL ignored good context.")
        print("    2. right doc, wrong section            -> a CHUNKING problem, not an")
        print("       embedding one. If faithfulness holds up here anyway, the neighbouring")
        print("       section carried the answer — and the retriever is better than 1. suggests.")
        print("    3. neither                             -> the RETRIEVER missed outright.")

    # 5. Safety — the one column where a low rate is the whole point.
    unsafe = df[df["safety_flag"]]
    print(f"\n  SAFETY — {len(unsafe)} of {len(df)} answers flagged as potentially harmful")
    if not unsafe.empty:
        by = (unsafe.groupby("producer").size().sort_values(ascending=False)
              .rename("flagged").to_frame())
        print("  " + by.to_string().replace("\n", "\n  "))

    print(f"\n  → {MASTER_PATH}\n")
