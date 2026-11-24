"""Paper tables from the run_ablations.sh summaries (Markdown for reading, LaTeX booktabs for the report).

    python make_tables.py outputs/ablations          # every machine under outputs/ablations/
    python make_tables.py outputs/ablations --machines rtx5070tilaptop a100

Expects <root>/<machine>/{baseline,edit_size,block_size}/summary.csv (missing pieces are skipped) and writes
<root>/tables.md and <root>/tables.tex. All values are means over prompts; "±" is the std over prompts.
Speedup is the latency of one CFG U-Net call on the edited latent relative to dense on the same machine.
"""

import argparse
import csv
import os

CFG_NAME = {"dense": "Dense SDXL", "sige_conv": "SIGE (conv)", "sige_conv_attn": "SIGE (conv+attn)"}
CFG_ORDER = ["dense", "sige_conv", "sige_conv_attn"]


def read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def num(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def fmt(row, key, digits=2, std=True, suffix=""):
    v = num(row, key)
    if v is None:
        return "--"
    s = "%.*f%s" % (digits, v, suffix)
    sd = num(row, key + "_std")
    if std and sd and int(row.get("n", 1)) > 1:
        s += " ± %.*f" % (digits, sd)
    return s


class Table:
    def __init__(self, title, caption, header, align):
        self.title, self.caption, self.header, self.align, self.rows = title, caption, header, align, []

    def add(self, cells, rule_before=False):
        self.rows.append((cells, rule_before))

    def markdown(self):
        out = ["### " + self.title, "", self.caption, "",
               "| " + " | ".join(self.header) + " |",
               "|" + "|".join("---:" if a == "r" else "---" for a in self.align) + "|"]
        out += ["| " + " | ".join(c) + " |" for c, _ in self.rows]
        return "\n".join(out) + "\n"

    def latex(self, label):
        esc = lambda s: (s.replace("%", r"\%").replace("±", r"$\pm$").replace("×", r"$\times$").replace("_", r"\_")
                         .replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")
                         .replace("**", "", 1).replace("**", "", 1))
        out = [r"\begin{table}[t]", r"\centering", r"\small",
               r"\caption{%s}" % esc(self.caption), r"\label{tab:%s}" % label,
               r"\begin{tabular}{%s}" % "".join(self.align), r"\toprule",
               " & ".join(esc(h) for h in self.header) + r" \\", r"\midrule"]
        for cells, rule in self.rows:
            if rule:
                out.append(r"\midrule")
            out.append(" & ".join(esc(c) for c in cells) + r" \\")
        out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(out) + "\n"


def by_cfg(rows):
    return {r["config"]: r for r in rows}


def baseline_table(machine, rows):
    t = Table("Baseline (%s)" % machine,
              "SIGE vs. dense SDXL at the base point (%s): 512 px, 10%% edit, 50 DDIM steps, strength 0.8, "
              "guidance 5.0, mean ± std over %s prompts. PSNR / LPIPS compare the full SDEdit output with dense "
              "SDXL given the same input, prompt, noise and seed." % (machine, rows[0]["n"]),
              ["Method", "GMACs", "MACs red.", "Latency (ms)", "Speedup", "PSNR (dB) ↑", "LPIPS ↓"],
              "lrrrrrr")
    cfgs = by_cfg(rows)
    for c in CFG_ORDER:
        if c in cfgs:
            r = cfgs[c]
            t.add([CFG_NAME[c], fmt(r, "gmacs", 0, False), fmt(r, "macs_reduction", 2, False, "×"),
                   fmt(r, "latency_ms", 1), fmt(r, "speedup", 2, True, "×"),
                   fmt(r, "psnr_vs_dense", 2), fmt(r, "lpips_vs_dense", 3)])
    return t


def edit_size_table(machine, rows):
    t = Table("Edit size (%s)" % machine,
              "Effect of edit size (%s, 512 px). Edit = measured fraction of changed pixels; MACs reduction and "
              "speedup are relative to dense SDXL; quality is vs. dense SDXL, mean ± std over %s prompts."
              % (machine, rows[0]["n"]),
              ["Edit", "Method", "GMACs", "MACs red.", "Speedup", "PSNR (dB) ↑", "LPIPS ↓"],
              "rlrrrrr")
    ratios = sorted({float(r["target_ratio"]) for r in rows})
    for i, ratio in enumerate(ratios):
        cfgs = by_cfg([r for r in rows if float(r["target_ratio"]) == ratio])
        first = True
        for c in CFG_ORDER[1:]:
            if c not in cfgs:
                continue
            r = cfgs[c]
            t.add(["%.1f%%" % (100 * num(r, "edit_ratio")) if first else "", CFG_NAME[c],
                   fmt(r, "gmacs", 0, False), fmt(r, "macs_reduction", 2, False, "×"),
                   fmt(r, "speedup", 2, True, "×"), fmt(r, "psnr_vs_dense", 2), fmt(r, "lpips_vs_dense", 3)],
                  rule_before=first and i > 0)
            first = False
    return t


def block_size_table(machine, rows):
    t = Table("Block size (%s)" % machine,
              "Effect of the SIGE block size (%s, 512 px, 10%% edit). Conv = block size of the 3×3 convolutions "
              "(resnet and upsample), Attn = token block size of the transformers; 1×1 shortcuts (4) and "
              "stride-2 downsamplers (5) are fixed. The attention block size does not affect SIGE (conv), whose "
              "transformers are dense. Defaults are 6 / 4." % machine,
              ["Conv", "Attn", "Method", "GMACs", "MACs red.", "Speedup", "PSNR (dB) ↑", "LPIPS ↓"],
              "rrlrrrrr")
    variants = []
    for r in rows:
        v = (int(r["conv_block_size"]), int(r["attn_block_size"]))
        if v not in variants:
            variants.append(v)
    variants.sort()
    for i, (conv, attn) in enumerate(variants):
        cfgs = by_cfg([r for r in rows if (int(r["conv_block_size"]), int(r["attn_block_size"])) == (conv, attn)])
        first = True
        for c in CFG_ORDER[1:]:
            if c not in cfgs:
                continue
            r = cfgs[c]
            tag = lambda x, d: ("**%d**" % x) if x == d else str(x)
            t.add([tag(conv, 6) if first else "", tag(attn, 4) if first else "", CFG_NAME[c],
                   fmt(r, "gmacs", 0, False), fmt(r, "macs_reduction", 2, False, "×"),
                   fmt(r, "speedup", 2, True, "×"), fmt(r, "psnr_vs_dense", 2), fmt(r, "lpips_vs_dense", 3)],
                  rule_before=first and i > 0)
            first = False
    return t


def machines_table(baselines):
    t = Table("Hardware comparison",
              "Baseline latency of one CFG U-Net call on each GPU (512 px, 10% edit, fp32). MACs are "
              "hardware independent; the speedup shows how much of the MACs reduction each GPU realizes.",
              ["GPU", "Dense (ms)", "SIGE conv (ms)", "SIGE conv+attn (ms)", "Speedup conv", "Speedup conv+attn",
               "MACs red. conv+attn"],
              "lrrrrrr")
    for machine, rows in baselines.items():
        cfgs = by_cfg(rows)
        g = lambda c, k, d, s=True, suf="": fmt(cfgs[c], k, d, s, suf) if c in cfgs else "--"
        t.add([machine, g("dense", "latency_ms", 1), g("sige_conv", "latency_ms", 1),
               g("sige_conv_attn", "latency_ms", 1), g("sige_conv", "speedup", 2, True, "×"),
               g("sige_conv_attn", "speedup", 2, True, "×"), g("sige_conv_attn", "macs_reduction", 2, False, "×")])
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="outputs/ablations")
    p.add_argument("--machines", nargs="+", default=None)
    args = p.parse_args()

    machines = args.machines or sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    tables, baselines = [], {}
    for m in machines:
        d = os.path.join(args.root, m)
        rows = read(os.path.join(d, "baseline", "summary.csv"))
        if rows:
            baselines[m] = rows
            tables.append(("baseline_" + m, baseline_table(m, rows)))
        rows = read(os.path.join(d, "edit_size", "summary.csv"))
        if rows:
            tables.append(("edit_size_" + m, edit_size_table(m, rows)))
        rows = read(os.path.join(d, "block_size", "summary.csv"))
        if rows:
            tables.append(("block_size_" + m, block_size_table(m, rows)))
    if len(baselines) > 1:
        tables.insert(0, ("machines", machines_table(baselines)))
    if not tables:
        raise SystemExit("No summary.csv files found under %s" % args.root)

    md = "# SIGE-SDXL ablation tables\n\n" + "\n".join(t.markdown() for _, t in tables)
    tex = "% Requires \\usepackage{booktabs}\n" + "\n".join(t.latex(label) for label, t in tables)
    with open(os.path.join(args.root, "tables.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(args.root, "tables.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    print(md.encode("ascii", "replace").decode() if os.name == "nt" else md)  # Windows consoles are cp1252
    print("Wrote %s and %s" % (os.path.join(args.root, "tables.md"), os.path.join(args.root, "tables.tex")))


if __name__ == "__main__":
    main()
