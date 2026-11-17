#!/usr/bin/env python
"""Audit of the pan-cancer report extraction: agreement between the bulk luna extraction (derived/report_pan/extracted_pan_all.jsonl)
and the independent Claude Sonnet extraction of 320 stratified reports (derived/report_pan/audit/extracted_sonnet_a*.jsonl).
Per field: agreement, agreement when at least one side reports a value, Cohen's kappa over the categorical values, and the
direction of the disagreements (luna only / Sonnet only / both differ). Writes results/collected/report_pan_audit.md and
derived/report_pan/audit/disagreements.csv (every disagreement with both quotes and the report context, for adjudication).
    python scripts/68_audit_pan_agreement.py"""
import glob, json, re
import numpy as np, pandas as pd
from opsis import paths
D = f"{paths.DERIVED}/report_pan"; AUD = f"{D}/audit"
schema = json.load(open(f"{paths.ROOT}/configs/reports/report_pan_schema.json")); fields = list(schema["fields"])
list_fields = [k for k, v in schema["fields"].items() if v.get("type") in ("list", "free text")]
rep = {}
for f in sorted(glob.glob(f"{AUD}/reports_a*.jsonl")):
    for j in map(json.loads, open(f)): rep[j["participant"]] = j
S = {}
for f in sorted(glob.glob(f"{AUD}/extracted_sonnet_a*.jsonl")):
    for j in map(json.loads, open(f)): S[j["participant"]] = j
L = {j["participant"]: j for j in map(json.loads, open(f"{D}/extracted_pan_all.jsonl")) if j["participant"] in rep}
P = sorted(set(rep) & set(S) & set(L)); print(f"reports in audit {len(rep)}, Sonnet {len(S)}, luna {len(L)}, aligned {len(P)}")

def val(c): return c if isinstance(c, list) else (c.get("value") if isinstance(c, dict) else c)
def quote(c):
    q = (c.get("quote") or "") if isinstance(c, dict) else ""
    return " | ".join(x for x in q if x) if isinstance(q, list) else q
def norm(v):
    if isinstance(v, list): return json.dumps(sorted(str(x.get("value") if isinstance(x, dict) else x).strip().lower() for x in v))
    return re.sub(r"\s+", " ", str(v)).strip().lower()
def ctx(p, q, w=90):
    t = rep[p]["text"]; i = t.find(q) if q else -1
    return ("…" + t[max(0, i - w): i + len(q) + w].replace("\n", " ") + "…") if i >= 0 else ""
def kappa(a, b):
    a, b = np.array(a), np.array(b); cats = sorted(set(a) | set(b)); n = len(a)
    if n == 0 or len(cats) < 2: return float("nan")
    po = float((a == b).mean()); pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)

rows = []; dis = []
for k in fields:
    va = [norm(val(L[p]["fields"].get(k, "not reported"))) for p in P]; vb = [norm(val(S[p]["fields"].get(k, "not reported"))) for p in P]
    eq = [x == y for x, y in zip(va, vb)]; nr = [x == "not reported" and y == "not reported" for x, y in zip(va, vb)]
    any_val = [not z for z in nr]; agree_val = sum(e for e, z in zip(eq, any_val) if z); n_val = sum(any_val)
    only_l = sum(1 for x, y in zip(va, vb) if x != y and y == "not reported"); only_s = sum(1 for x, y in zip(va, vb) if x != y and x == "not reported")
    both = sum(1 for x, y in zip(va, vb) if x != y and x != "not reported" and y != "not reported")
    n_both = sum(1 for x, y in zip(va, vb) if x != "not reported" and y != "not reported"); agree_both = n_both - both
    rows.append(dict(field=k, n=len(P), agree=f"{np.mean(eq):.3f}", n_any_value=n_val, agree_given_value=f"{agree_val / n_val:.3f}" if n_val else "", n_both_value=n_both, agree_both_value=f"{agree_both / n_both:.3f}" if n_both else "", kappa=f"{kappa(va, vb):.2f}" if k not in list_fields and k != "histologic_type" else "", luna_only=only_l, sonnet_only=only_s, both_differ=both))
    for p, x, y in zip(P, va, vb):
        if x != y:
            qa, qb = quote(L[p]["fields"].get(k, {})), quote(S[p]["fields"].get(k, {}))
            dis.append(dict(participant=p, project=rep[p]["project"], field=k, luna=x[:80], luna_quote=qa[:120], sonnet=y[:80], sonnet_quote=qb[:120], context=(ctx(p, qa.split(" | ")[0]) or ctx(p, qb.split(" | ")[0]))[:300]))
T = pd.DataFrame(rows); pd.DataFrame(dis).to_csv(f"{AUD}/disagreements.csv", index=False)
cl = sum(len(L[p].get("h_and_e_visible_claims", [])) for p in P); cs = sum(len(S[p].get("h_and_e_visible_claims", [])) for p in P)
tot = len(P) * len(fields); agree_tot = sum(int(r["agree"] is not None) for r in rows)
overall = np.mean([norm(val(L[p]["fields"].get(k, "not reported"))) == norm(val(S[p]["fields"].get(k, "not reported"))) for p in P for k in fields])
proj = pd.Series([rep[p]["project"] for p in P]).value_counts()
md = ["# Pan-cancer report extraction audit: luna bulk extraction vs an independent Claude Sonnet extraction", "",
      f"Reports audited: {len(P)} (stratified over {len(proj)} projects, {proj.min()} to {proj.max()} per project), {len(fields)} fields each, {tot} field comparisons.",
      f"Overall field agreement {overall:.3f}. Claims per report: luna {cl / len(P):.1f}, Sonnet {cs / len(P):.1f}.", "",
      "Columns: agree = fraction of reports where both extractions give the same value (not reported counts as a value); n_any_value = reports where at least one side reports a value; agree_given_value = agreement on those; n_both_value / agree_both_value = reports where both sides report a value and the agreement on those (free text fields are compared as exact strings, so their disagreements are mostly wording); kappa = Cohen's kappa over the categorical values (blank for free text and list fields); the last three columns split the disagreements by direction.", "",
      "| " + " | ".join(T.columns) + " |", "|" + "---|" * len(T.columns), *("| " + " | ".join(str(x) for x in r) + " |" for r in T.itertuples(index=False)), "", f"All {len(dis)} disagreements with both quotes and the report context are in `derived/report_pan/audit/disagreements.csv`."]
open(f"{paths.ROOT}/results/collected/report_pan_audit.md", "w").write("\n".join(md) + "\n"); print("\n".join(md[:4])); print(T.to_string(index=False))
