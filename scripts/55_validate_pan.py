#!/usr/bin/env python
"""Validate pan cancer extraction files against configs/reports/report_pan_schema.json: fields present, categorical
values allowed, quotes verbatim in the same participant's report, 1 to 8 claims, no negation words inside claims.
    python scripts/55_validate_pan.py --reports <jsonl> <extracted.jsonl> [<extracted2.jsonl> ...]"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from opsis import paths
ap = argparse.ArgumentParser(); ap.add_argument("--reports", required=True); ap.add_argument("--schema", default=f"{paths.ROOT}/configs/reports/report_pan_schema.json"); ap.add_argument("--repair", action="store_true", help="write <file>.repaired.jsonl with fuzzy matched quotes (ratio >= 0.9) and a _quote_flags list"); ap.add_argument("files", nargs="+"); a = ap.parse_args()
import difflib
def best_sub(t, q):
    L_ = len(q); best = (None, 0.0)
    for i in range(0, max(1, len(t) - L_), max(1, L_ // 3)):
        for w in (L_ - 5, L_, L_ + 5):
            cand = t[i:i + w]; r = difflib.SequenceMatcher(None, cand.lower(), q.lower()).ratio()
            if r > best[1]: best = (cand, r)
    return best
S = json.load(open(a.schema)); F = S["fields"]; texts = {j["participant"]: j["text"] for j in map(json.loads, open(a.reports))}
allowed = {k: set(x.strip() for x in v["values"].split("|")) for k, v in F.items() if v["type"] == "categorical"}
NEG = re.compile(r"(?i)\b(no|not|absent|without|negative|none|lacks?)\b")
for f in a.files:
    n = 0; problems = []; nr = 0; nq = 0; claims = 0; repaired = []
    for line in open(f):
        j = json.loads(line); n += 1; p = j.get("participant"); t = texts.get(p, "")
        if a.repair and "fields" in j:
            flags = []
            for k, cell in j["fields"].items():
                if not isinstance(cell, dict): continue
                qs = cell.get("quote"); islist = isinstance(qs, list)
                for idx, q in enumerate(qs if islist else [qs]):
                    if q and q not in t:
                        cand, r = best_sub(t, q)
                        if r >= 0.9:
                            if islist: cell["quote"][idx] = cand
                            else: cell["quote"] = cand
                            flags.append({"field": k, "ratio": round(r, 3), "original": q})
                        else: flags.append({"field": k, "ratio": round(r, 3), "original": q, "unrecoverable": True})
            if flags: j["_quote_flags"] = flags
            repaired.append(j)
        if "_error" in j: problems.append((p, "error", j["_error"][:80])); continue
        fl = j.get("fields", {})
        for k in F:
            if k not in fl: problems.append((p, k, "missing")); continue
            cell = fl[k]
            if isinstance(cell, list): cell = {"value": cell, "quote": ""}                       # bare list accepted for list fields
            elif not isinstance(cell, dict): cell = {"value": cell, "quote": ""}
            v = cell.get("value"); q = cell.get("quote", "") or ""
            if k in allowed and str(v) not in allowed[k]: problems.append((p, k, f"value {v!r} not allowed"))
            if isinstance(v, list):
                qlist = q if isinstance(q, list) else [q] * len(v)
                for it, qq in zip(v, qlist):
                    qq = it.get("quote", "") if isinstance(it, dict) else (qq or "")
                    if qq and qq not in t: problems.append((p, k, "list quote not verbatim"))
                continue
            if v == "not reported": nr += 1
            if q:
                nq += 1
                if q not in t:
                    ws = lambda x: re.sub(r"\s+", " ", x)
                    problems.append((p, k, ("quote verbatim only after whitespace normalisation: " if ws(q) in ws(t) else "quote not verbatim: ") + repr(q[:50])))
            elif v not in ("not reported", "", None): problems.append((p, k, f"value {str(v)[:30]!r} without quote"))
        cl = j.get("h_and_e_visible_claims", [])
        if not (1 <= len(cl) <= 8): problems.append((p, "claims", f"{len(cl)} claims"))
        for c in cl:
            claims += 1
            if c.get("polarity") not in ("present", "absent"): problems.append((p, "claims", "bad polarity"))
            if NEG.search(c.get("claim", "")): problems.append((p, "claims", f"negation inside claim: {c.get('claim','')[:50]!r}"))
    if a.repair:
        out = f[:-6] + ".repaired.jsonl" if f.endswith(".jsonl") else f + ".repaired.jsonl"
        with open(out, "w") as fo:
            for j in repaired: fo.write(json.dumps(j, ensure_ascii=False) + "\n")
        nf = sum(len(j.get("_quote_flags", [])) for j in repaired); nu = sum(1 for j in repaired for fl in j.get("_quote_flags", []) if fl.get("unrecoverable"))
        print(f"  repaired copy {out}: {nf} quotes fuzzy matched, {nu} unrecoverable (flagged)")
    print(f"{f.split('/')[-1]}: {n} records | fields not reported {nr} | quoted values {nq} | claims {claims} | problems {len(problems)}")
    for pr in problems[:25]: print("   ", pr)
