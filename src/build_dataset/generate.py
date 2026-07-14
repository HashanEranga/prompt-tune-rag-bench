"""Stage 3 — draft grounded Q&A pairs from FAQ sections + fact tables.

Produces ``data/qa/pool.jsonl`` and a human ``verify_sheet.md``. Hand-authored
fine-grained pairs are merged in from an ``authored_pairs`` module (expected at
``src/authored_pairs.py``); if that module is absent this stage raises
ImportError — add the file, or comment out the merge, before running.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .config import QA_DIR, ROOT
from .segment import iter_sections, parse_clean

TOPIC = {
    "SGH-SU-001": "surgery-pricing", "SGH-OPD-001": "opd-services",
    "SGH-LAB-001": "lab-services", "SGH-FIN-001": "insurance-billing",
    "SGH-ER-001": "emergency", "SGH-PH-001": "pharmacy", "SGH-IP-001": "inpatient",
    "SGH-PI-001": "general-info", "SGH-FAQ-002": "appointments",
    "SGH-FAQ-003": "surgery", "SGH-FAQ-004": "inpatient",
    "SGH-FAQ-005": "lab", "SGH-FAQ-006": "insurance",
}


def _table_to_text(tbl: dict) -> str:
    lines = [" | ".join(tbl["header"])] + [" | ".join(r) for r in tbl["rows"]]
    return " ;; ".join(lines)


def _classify(text: str) -> str:
    if re.search(r"LKR|\d{1,3}(?:,\d{3})+|\b\d+\s*%|\b\d+\s*(?:hours?|minutes?|days?|min)\b"
                 r"|\d{1,2}:\d{2}\s*[ap]m", text, re.I):
        return "numeric"
    if re.search(r"\bmust\b|\brequired\b|\bshould\b|\bbring\b|\bapply\b|\bcontact\b|\bsubmit\b",
                 text, re.I):
        return "procedural"
    return "factual"


def _needs_attention(answer: str, evidence: str) -> bool:
    """Flag if a number asserted in the answer is not literally in the evidence."""
    ev = evidence.replace(",", "")
    return any(n.replace(",", "") not in ev for n in re.findall(r"\d[\d,]*", answer))


def _pair(fm, clean_path, counter, question, answer, evidence, section, atype=None):
    did = fm["doc_id"]
    counter[did] = counter.get(did, 0) + 1
    answer = answer.strip()
    evidence = evidence.strip()
    return {
        "id": f"{did.lower()}-q{counter[did]:03d}",
        "question": question.strip(),
        "answer": answer,
        "source_doc": fm.get("source_pdf", ""),
        "clean_path": str(Path(clean_path).relative_to(ROOT)) if Path(clean_path).is_absolute() else str(clean_path),
        "doc_id": did,
        "section": section,
        "evidence": evidence,
        "topic": TOPIC.get(did, "general"),
        "answer_type": atype or _classify(answer),
        "needs_attention": _needs_attention(answer, evidence),
        "verified": False,
        "verify_note": "",
    }


def _faq_pairs(fm, clean_path, blocks, counter):
    """Each ### question + its following prose answer becomes a grounded pair."""
    pairs, parent = [], ""
    for idx, b in enumerate(blocks):
        if b["kind"] == "heading" and b["level"] == 2:
            parent = b["text"]
        if b["kind"] == "heading" and b["level"] == 3 and b["text"].rstrip().endswith("?"):
            ans = []
            for nb in blocks[idx + 1:]:
                if nb["kind"] == "heading":
                    break
                if nb["kind"] == "para":
                    ans.append(nb["text"])
                elif nb["kind"] == "bullets":
                    ans.append("; ".join(nb["items"]))
                elif nb["kind"] == "table":
                    ans.append(_table_to_text(nb))
            answer = " ".join(ans).strip()
            if answer:
                pairs.append(_pair(fm, clean_path, counter, b["text"], answer, answer, parent))
    return pairs


def _fee(v: str) -> str:
    v = v.strip()
    if v and v[0].isdigit() and "%" not in v:
        return f"{v} LKR"
    return v


def _table_pairs(fm, clean_path, section, tbl, counter):
    """Dispatch a fact table to natural, grounded Q&A templates by its columns."""
    H = tbl["header"]
    hset = " | ".join(H)
    pairs = []

    def mk(q, a, ev, atype=None):
        pairs.append(_pair(fm, clean_path, counter, q, a, ev, section, atype))

    # The ward deposit table's header row was mis-parsed into the first data row, so
    # iterate [header] + rows with the known Ward / Deposit / Inclusions columns.
    if re.search(r"\b(Ward|Room|Suite)\b", H[0]) and len(H) >= 3 and re.fullmatch(r"[\d,]+", H[1]):
        for row in [H] + tbl["rows"]:
            ward, dep, incl = (row + ["", "", ""])[:3]
            ev = " | ".join(row)
            mk(f"What is the admission deposit for a {ward} at Serendib General Hospital?",
               f"{dep} LKR.", ev, "numeric")
            if incl:
                mk(f"What is included with a {ward} at Serendib General Hospital?", f"{incl}.", ev)
        return pairs

    if H[:1] == ["Ordinarily Covered"] and len(H) >= 2:  # insurance coverage lists
        ev = _table_to_text(tbl)
        covered = "; ".join(r[0] for r in tbl["rows"] if r and r[0])
        excluded = "; ".join(r[1] for r in tbl["rows"] if len(r) > 1 and r[1])
        mk("What is ordinarily covered under insurance at Serendib General Hospital?", f"{covered}.", ev)
        mk("What is ordinarily excluded from insurance coverage at Serendib General Hospital?",
           f"{excluded}.", ev)
        return pairs

    if "Emergency Hotline" in H and tbl["rows"]:  # values stacked in two columns
        ev = _table_to_text(tbl)
        r0 = tbl["rows"][0]
        note = tbl["rows"][1] if len(tbl["rows"]) > 1 else ["", ""]
        mk("What is the emergency hotline number for Serendib General Hospital?",
           f"{r0[0]} (available {note[0]}).", ev, "numeric")
        if len(r0) > 1:
            mk("What is the ambulance dispatch number at Serendib General Hospital?",
               f"{r0[1]} ({note[1] if len(note) > 1 else ''}).".replace(" ().", "."), ev, "numeric")
        return pairs

    def add(q, a, atype=None):
        mk(q, a, " | ".join(row), atype)

    for row in tbl["rows"]:
        if len(row) < len(H):
            row = row + [""] * (len(H) - len(row))

        if "Procedure" in H and any("Estimated Cost" in h for h in H):
            proc, spec, cost, stay = row[:4]
            add(f"What is the estimated cost of {proc} at Serendib General Hospital?",
                f"The estimated cost is {cost} (LKR).", atype="numeric")
            add(f"What is the typical hospital stay for {proc} at Serendib?", f"{stay}.")
            add(f"Which surgical specialty performs {proc} at Serendib General Hospital?", f"{spec}.")

        elif "Specialty" in H and any("Wait" in h for h in H):
            spec, days, hours, wait = row[:4]
            add(f"On which days is the {spec} clinic held at the Serendib OPD?", f"{days}.")
            add(f"What are the OPD consultation hours for {spec} at Serendib?", f"{hours}.", atype="numeric")
            add(f"What is the estimated waiting time for a booked {spec} appointment at Serendib?",
                f"{wait}.", atype="numeric")

        elif hset.startswith("Category | Fee") or (H[:1] == ["Category"] and "Fee (LKR)" in H):
            cat, fee = row[0], row[1]
            add(f"What is the consultation fee for a {cat} at Serendib General Hospital?", f"{_fee(fee)}.",
                atype="numeric")

        elif "Examples" in H and "Turnaround" in H:
            cat, ex, ta = row[:3]
            add(f"What is the turnaround time for {cat} tests at the Serendib laboratory?", f"{ta}.",
                atype="numeric")
            add(f"Which tests are included under {cat} at Serendib's laboratory?", f"{ex}.")

        elif "Modality" in H:
            mod, prep, ta = row[:3]
            add(f"What preparation is required before a {mod} at Serendib?", f"{prep}.", atype="procedural")
            add(f"How long does the {mod} report take at Serendib General Hospital?", f"{ta}.", atype="numeric")

        elif "Fasting Required" in H:
            test, fasting, dur = row[:3]
            a = f"{fasting} — {dur}." if fasting.lower().startswith("yes") else f"{fasting}."
            add(f"Is fasting required for a {test} at Serendib General Hospital?", a)

        elif "Insurer" in H:
            ins, direct, pre = row[:3]
            add(f"Does Serendib General Hospital offer direct billing with {ins}?",
                f"{direct}. Pre-authorisation: {pre}.")

        elif "Method" in H and any("Available" in h for h in H):
            method, at, notes = (row + ["", "", ""])[:3]
            a = f"Yes, at {at}."
            if notes and notes.lower() != "none":
                a += f" Note: {notes}."
            add(f"Can I pay by {method} at Serendib General Hospital, and where?", a)

        elif "Level" in H and any("Target" in h for h in H):
            lvl, cat, desc, target = row[:4]
            add(f"At Serendib's emergency department, what does Triage Level {lvl} ({cat}) mean, "
                f"and how quickly should the patient be seen by a doctor?",
                f"{desc}. Target: seen {target.lower()}.", atype="numeric")

        elif hset == "Item | Fee (LKR)":
            item, fee = row[:2]
            add(f"What is the fee for {item} at Serendib's emergency department?", f"{_fee(fee)}.",
                atype="numeric")

        elif "Pharmacy" in H and "Location" in H:
            ph, loc, hrs = row[:3]
            add(f"Where is the {ph} at Serendib General Hospital and what are its hours?",
                f"{loc}; open {hrs}.")

        elif "Request Type" in H:
            req, ta = row[:2]
            add(f"How long does a {req} take at the Serendib pharmacy?", f"{ta}.", atype="numeric")

        else:  # generic key -> value fallback
            key = row[0]
            for j in range(1, len(H)):
                if H[j] and row[j]:
                    add(f"For {key} at Serendib General Hospital, what is the {H[j]}?", f"{row[j]}.")

    return pairs


# Natural question stems for the prose sections of the non-FAQ docs.
PROSE_QSTEM = {
    "ABOUT SERENDIB GENERAL HOSPITAL": "What is Serendib General Hospital — its size, location, and accreditation?",
    "Visiting Hours": "What are the visiting hours at Serendib General Hospital?",
    "Appointment Booking": "How can I book an appointment at Serendib General Hospital?",
    "Languages": "Which languages and translation services are available at Serendib General Hospital?",
    "Parking": "What parking options and charges are available at Serendib General Hospital?",
    "24-Hour Pharmacy": "Does Serendib General Hospital have a 24-hour pharmacy, and where is it?",
    "Medical Records": "How can I obtain a copy of my medical records from Serendib General Hospital?",
    "Second Opinions": "How can I get a second opinion at Serendib General Hospital?",
    "Facilities for Patients with Disabilities": "What facilities does Serendib General Hospital provide for patients with disabilities?",
    "Feedback and Complaints": "How can I submit feedback or a complaint to Serendib General Hospital?",
    "EMERGENCY ENTRANCE AND ACCESS": "Where is the emergency entrance at Serendib General Hospital and how is it accessed?",
    "CONDITIONS REQUIRING IMMEDIATE AEC ATTENDANCE": "Which conditions require immediate emergency (AEC) attendance at Serendib General Hospital?",
    "PAEDIATRIC EMERGENCY": "How does the paediatric emergency service work at Serendib General Hospital?",
    "ADMISSION PROCESS": "What is the inpatient admission process at Serendib General Hospital?",
    "ITEMS TO BRING FOR ADMISSION": "What items should a patient bring for admission to Serendib General Hospital?",
    "DURING YOUR STAY": "What should inpatients know about their stay at Serendib General Hospital?",
    "DISCHARGE": "What is the discharge process for inpatients at Serendib General Hospital?",
    "VISITOR POLICY": "What is the visitor policy for inpatients at Serendib General Hospital?",
    "PRE-AUTHORISATION": "How does insurance pre-authorisation work for admissions at Serendib General Hospital?",
    "ITEMISED BILLING": "How are patient bills itemised at Serendib General Hospital?",
    "FINANCIAL ASSISTANCE": "What financial assistance or payment plans does Serendib General Hospital offer?",
    "COLLECTING RESULTS": "How can patients collect their laboratory or imaging results from Serendib General Hospital?",
    "OVERVIEW": "What are the Serendib OPD's opening hours and approximate daily patient volume?",
    "WALK-IN PATIENTS": "How are walk-in patients handled at the Serendib OPD?",
    "CHANNELLING SYSTEM": "How does the channelling (queue) system work at Serendib General Hospital?",
    "DOCUMENTS TO BRING TO OPD": "What documents should a patient bring to the Serendib OPD?",
    "PRESCRIPTION DISPENSING": "How are prescriptions dispensed at the Serendib pharmacy?",
    "MEDICATION AVAILABILITY": "How large is the medication formulary at Serendib General Hospital?",
    "GENERIC SUBSTITUTION POLICY": "What is Serendib General Hospital's generic substitution policy?",
    "CONTROLLED SUBSTANCES": "How are controlled substances dispensed at Serendib General Hospital?",
    "HOME DELIVERY SERVICE": "Does Serendib General Hospital offer prescription home delivery, and how does it work?",
    "DRUG INFORMATION HELPLINE": "Is there a drug information helpline at Serendib General Hospital?",
    "PRE-OPERATIVE REQUIREMENTS": "What are the pre-operative requirements before surgery at Serendib General Hospital?",
}

# Per-(doc, heading) overrides where the same heading appears in two docs with
# different content (avoids an identical question mapping to two answers).
PROSE_QSTEM_OVERRIDE = {
    ("SGH-SU-001", "ITEMS TO BRING FOR ADMISSION"):
        "What items should a patient bring for a surgical admission at Serendib General Hospital?",
}


def _prose_pairs(fm, clean_path, blocks, counter):
    """One grounded pair per prose section, using an authored question stem."""
    pairs = []
    for sec in iter_sections(blocks):
        parts = []
        for b in sec["content"]:
            if b["kind"] == "para":
                parts.append(b["text"])
            elif b["kind"] == "bullets":
                parts.append("; ".join(b["items"]))
        answer = " ".join(parts).strip()
        if len(answer) < 40:
            continue
        q = PROSE_QSTEM_OVERRIDE.get((fm["doc_id"], sec["heading"])) or PROSE_QSTEM.get(sec["heading"])
        if not q:  # skip sections without an authored stem (e.g. surgical items dup)
            continue
        pairs.append(_pair(fm, clean_path, counter, q, answer, answer, sec["parent"] or sec["heading"]))
    return pairs


def cmd_generate(_args) -> None:
    from .config import CLEAN_DIR

    QA_DIR.mkdir(parents=True, exist_ok=True)
    counter: dict = {}
    pool = []
    fm_by_did: dict = {}
    for md in sorted(CLEAN_DIR.glob("*.md")):
        fm, blocks = parse_clean(md)
        did = fm.get("doc_id", "")
        fm_by_did[did] = (fm, md)
        if did.startswith("SGH-FAQ"):
            pool += _faq_pairs(fm, md, blocks, counter)
        else:
            pool += _prose_pairs(fm, md, blocks, counter)
        for sec in iter_sections(blocks):
            for b in sec["content"]:
                if b["kind"] == "table":
                    pool += _table_pairs(fm, md, sec["heading"], b, counter)

    # authored_pairs.py lives in src/, one level above this package.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from authored_pairs import AUTHORED
    n_auth = 0
    for did, section, q, a, ev, atype in AUTHORED:
        if did not in fm_by_did:
            print(f"  ! authored pair references unknown doc {did}, skipping")
            continue
        fm, md = fm_by_did[did]
        pool.append(_pair(fm, md, counter, q, a, ev, section, atype))
        n_auth += 1

    out = QA_DIR / "pool.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for rec in pool:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _write_verify_sheet(pool)

    by_doc: dict = {}
    for r in pool:
        by_doc[r["doc_id"]] = by_doc.get(r["doc_id"], 0) + 1
    na = sum(r["needs_attention"] for r in pool)
    print(f"Generated {len(pool)} grounded Q&A pairs "
          f"({len(pool) - n_auth} auto + {n_auth} authored; {na} need attention) "
          f"-> {out.relative_to(ROOT)}")
    print(f"Review sheet -> {(QA_DIR / 'verify_sheet.md').relative_to(ROOT)}")
    for d in sorted(by_doc):
        print(f"  {d:12} {by_doc[d]:3}")


def _write_verify_sheet(pool: list[dict]) -> None:
    """Human review checklist grouped by doc; needs_attention pairs surfaced first."""
    lines = [
        "# Q&A verification sheet",
        "",
        "Verify every pair **against its source document** (`data/clean/<doc>.md`, "
        "and the original PDF when in doubt) — the source is the authority, not the "
        "drafted answer. Tick the box when confirmed; note any correction inline.",
        "",
        f"- Total pairs: **{len(pool)}**",
        f"- Flagged `needs_attention` (a number in the answer wasn't found verbatim in "
        f"the evidence): **{sum(r['needs_attention'] for r in pool)}**",
        "",
    ]
    by_doc: dict = {}
    for r in pool:
        by_doc.setdefault(r["doc_id"], []).append(r)
    for did in sorted(by_doc):
        rows = sorted(by_doc[did], key=lambda r: (not r["needs_attention"], r["id"]))
        lines.append(f"## {did} — {rows[0]['source_doc']}  ({len(rows)} pairs)")
        lines.append("")
        for r in rows:
            flag = " ⚠️" if r["needs_attention"] else ""
            lines.append(f"- [ ] **{r['id']}**{flag} · _{r['section']}_ · `{r['answer_type']}`")
            lines.append(f"  - **Q:** {r['question']}")
            lines.append(f"  - **A:** {r['answer']}")
            lines.append(f"  - **Evidence:** {r['evidence']}")
        lines.append("")
    (QA_DIR / "verify_sheet.md").write_text("\n".join(lines), encoding="utf-8")
