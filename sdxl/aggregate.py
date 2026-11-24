"""Average benchmark.py results.csv files (e.g. one per prompt) per (resolution, target_ratio, config).

    python aggregate.py outputs/edit_size/prompt*/results.csv --out outputs/edit_size/summary.csv
"""

import argparse
import csv
from collections import defaultdict

import numpy as np

METRICS = ["edit_ratio", "gmacs", "latency_ms", "macs_reduction", "speedup", "psnr_vs_dense", "lpips_vs_dense"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csvs", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    groups = defaultdict(lambda: defaultdict(list))
    for path in args.csvs:
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (int(row["resolution"]), float(row["target_ratio"]), row["config"])
                for m in METRICS:
                    if row.get(m):
                        groups[key][m].append(float(row[m]))

    order = {"dense": 0, "sige_conv": 1, "sige_conv_attn": 2}
    fields = ["resolution", "target_ratio", "config", "n"]
    for m in METRICS:
        fields += [m, m + "_std"]
    rows = []
    for (res, ratio, cfg), vals in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], order.get(kv[0][2], 9))):
        row = {"resolution": res, "target_ratio": ratio, "config": cfg, "n": len(vals["gmacs"])}
        for m in METRICS:
            if vals[m]:
                row[m] = "%.4g" % np.mean(vals[m])
                row[m + "_std"] = "%.3g" % np.std(vals[m])
        rows.append(row)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("  ".join("%s=%s" % (k, r.get(k, "")) for k in
                        ["target_ratio", "config", "n", "macs_reduction", "speedup", "psnr_vs_dense", "lpips_vs_dense"]))
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
