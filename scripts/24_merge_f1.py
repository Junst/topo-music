"""Merge the per-arm probe_f1_<arm>.json files that the array writes."""
import json
from pathlib import Path

out = {}
for f in sorted(Path("runs").glob("probe_f1_*.json")):
    out.update(json.loads(f.read_text()))
Path("runs/probe_f1.json").write_text(json.dumps(out, indent=2))
hdr = f"{'arm':<12}{'F1_grp':>8}{'acc_grp':>9}{'F1_shuf':>9}{'leak':>7}"
print(hdr); print("-" * len(hdr))
for a, r in sorted(out.items(), key=lambda kv: -kv[1]["grouped"]["macro_f1"]):
    g, s = r["grouped"], r["shuffled"]
    print(f"{a:<12}{g['macro_f1']:>8.3f}{g['acc']:>9.3f}{s['macro_f1']:>9.3f}"
          f"{s['acc'] - g['acc']:>+7.3f}")
print(f"\n{len(out)} arms -> runs/probe_f1.json")
