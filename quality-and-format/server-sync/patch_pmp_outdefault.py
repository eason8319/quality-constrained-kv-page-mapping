from pathlib import Path
p = Path("experiments/page_paired_sensitivity/protect_multipage.py")
text = p.read_text()
old = (
    '    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "runs" / "protect_multipage_v1"\n'
    '    if args.smoke and args.out_dir is None:\n'
    '        out_dir = ROOT / "runs" / "protect_multipage_v1_smoke"'
)
new = (
    '    if args.out_dir:\n'
    '        out_dir = Path(args.out_dir)\n'
    '    elif args.template == "nonretr":\n'
    '        out_dir = ROOT / "runs" / "protect_multipage_nonretr_v1"\n'
    '    else:\n'
    '        out_dir = ROOT / "runs" / "protect_multipage_v1"\n'
    '    if args.smoke and args.out_dir is None:\n'
    '        out_dir = ROOT / "runs" / (\n'
    '            "protect_multipage_nonretr_v1_smoke" if args.template == "nonretr" else "protect_multipage_v1_smoke"\n'
    '        )'
)
if old not in text:
    raise SystemExit("patch anchor missing")
p.write_text(text.replace(old, new, 1))
print("ok")
