#!/usr/bin/env python
"""Report extraction through an OpenAI compatible chat API (key read from a file, never printed): one JSON object per
report following a schema file, with the same rules as the Claude subagent pipeline (verbatim quotes, never infer,
never pad). Used for the model comparison pilot and for the pan cancer bulk extraction (scripts/54_run_pan_bulk.sh).
    python scripts/54_extract_openai.py --reports <jsonl> --schema <json> --out <jsonl> [--model gpt-6-luna] [--workers 4]"""
import argparse, json, os, ssl, time, urllib.request, concurrent.futures as cf
import certifi
from opsis.keys import api_key

ap = argparse.ArgumentParser()
ap.add_argument("--reports", required=True); ap.add_argument("--schema", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="gpt-6-luna"); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--key-file", default=None, help="file holding the API key (default: $OPENAI_API_KEY)")
ap.add_argument("--base", default="https://api.openai.com/v1"); ap.add_argument("--max-tokens", type=int, default=4000); ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
KEY = api_key(a.key_file); CTX = ssl.create_default_context(cafile=certifi.where())
schema = json.load(open(a.schema)); schema_text = json.dumps(schema, indent=1)
SYSTEM = ("You extract structured facts from pathology reports for a research dataset. Follow the schema exactly. Rules: use only what the report "
          "states, never infer; use exactly the categorical values listed; when a field is not stated set value to \"not reported\" and quote to \"\"; "
          "every non empty quote must be a short verbatim substring of the report text (copy characters exactly, including typos); standard "
          "unambiguous conventions may be translated as the schema describes; hedged statements stay not reported; follow the specimen_rule; "
          "claims are finding phrases with the polarity field, 1 to 8 of them, never padded. Copy every quote character for character from the report, including line breaks and periods that the report has inside the phrase, so that it is an exact substring. Record multifocal, precursor lesions and tumour cell percentage only as the schema defines them, never by inference. Every field, including list fields, must be an object with value and quote. Output a single JSON object and nothing else.")
def call(rec):
    body = {"model": a.model, "max_completion_tokens": a.max_tokens, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM + "\n\nSCHEMA:\n" + schema_text},
                         {"role": "user", "content": f"participant: {rec['participant']}\nproject: {rec.get('project', '')}\n\nREPORT:\n{rec['text']}\n\nReturn the JSON object described by the schema's \"output\" entry."}]}
    req = urllib.request.Request(a.base + "/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=300, context=CTX) as r: d = json.load(r)
            txt = d["choices"][0]["message"]["content"]; u = d.get("usage", {})
            obj = json.loads(txt); obj["participant"] = rec["participant"]; obj.setdefault("project", rec.get("project", ""))
            obj["_usage"] = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"), "seconds": round(time.time() - t0, 1), "model": d.get("model", a.model)}
            return obj
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            if hasattr(e, "read"):
                try: err += " | " + e.read().decode()[:300]
                except Exception: pass
            time.sleep(5 * (attempt + 1))
    return {"participant": rec["participant"], "project": rec.get("project", ""), "_error": err}
recs = [json.loads(l) for l in open(a.reports)]
if a.limit: recs = recs[:a.limit]
with cf.ThreadPoolExecutor(a.workers) as ex: outs = list(ex.map(call, recs))
with open(a.out, "w") as f:
    for o in outs: f.write(json.dumps(o) + "\n")
ok = [o for o in outs if "_error" not in o]; pt = sum(o["_usage"]["prompt_tokens"] or 0 for o in ok); ct = sum(o["_usage"]["completion_tokens"] or 0 for o in ok)
print(f"EXTRACT_DONE {len(ok)}/{len(outs)} ok | prompt tokens {pt} | completion tokens {ct} | errors: {[o['_error'] for o in outs if '_error' in o][:3]}")
