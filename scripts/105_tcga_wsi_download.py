#!/usr/bin/env python
"""Download TCGA slides listed by scripts/104 from the GDC data endpoint into $OPSIS_WSI with the layout the code
expects (TCGA-<project>/<file uuid>/<slide>.svs, the gdc-client layout). curl per file with resume
(-C -), size and md5 check against the manifest, N parallel streams, a pid file for stopping (kill $(cat <out>/manifests/download.pid)),
rerun to resume: finished files are skipped, partial files continue. Open access slides only (diagnostic TCGA slides are open).
    PYTHONPATH=src python scripts/105_tcga_wsi_download.py --tiers 1 [--workers 4] [--limit 2] [--dry-run]
    PYTHONPATH=src python scripts/105_tcga_wsi_download.py --tiers 1,2,3,4 --slides <file> --workers 6     # a named subset, e.g. the 127 BRCA bench slides outside the test split"""
import argparse, hashlib, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from opsis import paths
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=paths.WSI); ap.add_argument("--tiers", default="1"); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--limit", type=int, default=None); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--no-md5", action="store_true"); ap.add_argument("--projects", default=None); ap.add_argument("--slides", default=None, help="file with one slide name per line: download only these (any tier given by --tiers)")
a = ap.parse_args(); M = f"{a.out}/manifests"; URL = "https://api.gdc.cancer.gov/data/"
S = pd.read_csv(f"{M}/all.csv"); S = S[S.file_id.notna() & S.tier.isin([int(t) for t in a.tiers.split(",")]) & (S.access == "open")]
if a.projects: S = S[S.project.isin(a.projects.split(","))]
if a.slides: S = S[S.slide.isin({l.strip() for l in open(a.slides) if l.strip()})]
S = S.sort_values(["tier", "project", "slide"]).reset_index(drop=True)
if a.limit: S = S.head(a.limit)
def dest(r): return f"{a.out}/TCGA-{r.project}/{r.file_id}/{r.slide}.svs"
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def done(r):
    p = dest(r); return os.path.exists(p) and os.path.getsize(p) == int(r.size)
todo = [r for r in S.itertuples() if not done(r)]
print(f"{len(S)} slides in tiers {a.tiers}, {len(S) - len(todo)} already complete, {len(todo)} to download, {sum(r.size for r in todo) / 1e12:.2f} TB", flush=True)
if a.dry_run: sys.exit(0)
open(f"{M}/download.pid", "w").write(str(os.getpid()))
def fetch(r):
    p = dest(r); part = p + ".part"; os.makedirs(os.path.dirname(p), exist_ok=True); t0 = time.time()
    for attempt in range(4):
        cmd = ["curl", "-sS", "-L", "--retry", "5", "--retry-delay", "10", "--speed-limit", "10000", "--speed-time", "120", "-C", "-", "-o", part, URL + r.file_id]
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        if rc == 33 or (rc == 0 and os.path.exists(part) and os.path.getsize(part) != int(r.size)):      # 33 = range not satisfiable (server has no resume) -> restart clean
            if os.path.exists(part) and os.path.getsize(part) > int(r.size): os.remove(part)
            if rc == 33: os.remove(part) if os.path.exists(part) else None; continue
        if rc == 0 and os.path.exists(part) and os.path.getsize(part) == int(r.size):
            if not a.no_md5 and r.md5 and md5(part) != r.md5: os.remove(part); print(f"MD5 MISMATCH {r.slide} attempt {attempt}", flush=True); continue
            os.replace(part, p); return f"ok {r.slide} {int(r.size) / 1e9:.2f} GB {time.time() - t0:.0f} s"
        time.sleep(10 * (attempt + 1))
    return f"FAILED {r.slide} rc {rc} size {os.path.getsize(part) if os.path.exists(part) else 0}/{r.size}"
t00 = time.time(); n = 0; fails = []
with ThreadPoolExecutor(a.workers) as ex:
    for line in ex.map(fetch, todo):
        n += 1; print(f"[{n}/{len(todo)}] {line} ({(time.time() - t00) / 60:.0f} min)", flush=True)
        if line.startswith("FAILED"): fails.append(line)
open(f"{M}/failed_{a.tiers.replace(',', '')}.txt", "w").write("\n".join(fails) + ("\n" if fails else ""))
print(f"done: {n - len(fails)} downloaded, {len(fails)} failed, {(time.time() - t00) / 3600:.1f} h", flush=True)
if os.path.exists(f"{M}/download.pid"): os.remove(f"{M}/download.pid")
