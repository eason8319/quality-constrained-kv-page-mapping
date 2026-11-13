"""Materialize one official Llama D_cal RULER input per task/length, and WikiText segment IDs."""
import argparse
import csv
import hashlib
import json
import os
import runpy
import sys
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

root = Path.cwd()
run = Path(__file__).resolve().parent
raw = root / "data/raw"
source = raw/"sources"
split = root/"data/splits"
ruler = source/"ruler"
folder = raw/"p0a-ruler"
folder.mkdir(exist_ok=True)
audit = json.loads((run/"split_audit.json").read_bytes())
cfg = yaml.safe_load((root/"configs/data_manifest.yaml").read_bytes())
args = argparse.ArgumentParser()
args.add_argument("part", choices=["wiki","ruler"])
part = args.parse_args().part


def write_ids(task, group, ids):
    dest = split/task/(group+".txt")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(("\n".join(ids) + ("\n" if ids else "")).encode())


def wiki():
    # Reconstruct the article exactly as indexed in make_splits, then verify its raw hash.
    import re
    docs = []
    file, = (source/"wikitext/wikitext-2-raw-v1").glob("validation*.parquet")
    for r in pq.read_table(file).to_pylist():
        s = r["text"]
        if re.fullmatch(r"= (?![= ])[^=].*? =",s.strip()) and not s.strip().endswith("= ="):
            docs.append([])
        if docs:
            docs[-1].append(s)
    with (run/"source_index.tsv").open() as f:
        rows = {r["sample_id"]:r for r in csv.DictReader(f,delimiter="\t") if r["task"]=="wikitext2"}
    index=[]
    for spec in yaml.safe_load((root/"configs/models.yaml").read_bytes())["models"]:
        model = spec["short"]
        tok = AutoTokenizer.from_pretrained(source/model,local_files_only=True)
        ids = {s:[] for s in ("D_cal","D_sel","D_test")}
        for n,lines in enumerate(docs):
            text = "\n".join(lines)
            row = rows[f"wikitext2/validation/article/{n}"]
            assert row["raw_sha256"] == hashlib.sha256(text.encode()).hexdigest()
            tokens = tok.encode(text,add_special_tokens=False)
            for offset in range(0,len(tokens)-2047,2048):
                sample = f"{row['sample_id']}/{model}/{offset}"
                ids[row["split"]].append(sample)
                block = tokens[offset:offset+2048]
                digest = hashlib.sha256(b"".join(t.to_bytes(4,"big") for t in block)).hexdigest()
                index.append([sample,row["sample_id"],model,spec["revision"],row["split"],offset,2048,digest])
        for group,entries in ids.items():
            write_ids("wikitext2_segments/"+model,group,entries)
    with (run/"wikitext_segments.tsv").open("w",newline="") as f:
        w=csv.writer(f,delimiter="\t",lineterminator="\n")
        w.writerow(["sample_id","article","model","revision","split","offset","count","u32be_sha256"])
        w.writerows(index)


def build_ruler():
    rev = next(c["revision"] for c in cfg["datasets"] if c["task"]=="ruler")
    with tarfile.open(ruler/(rev+".tar.gz")) as tar:
        tar.extractall(folder,filter="data")
    scripts = folder/("RULER-"+rev)/"scripts"
    assets = scripts/"data/synthetic/json"
    word_file = assets/"english_words.json"
    pointer = word_file.read_text().splitlines()
    words = (ruler/"english_words.json").read_bytes()
    assert "oid sha256:"+hashlib.sha256(words).hexdigest() in pointer
    word_file.write_bytes(words)
    (assets/"PaulGrahamEssays.json").write_bytes((ruler/"PaulGrahamEssays.json").read_bytes())
    qa_ids = {}
    for sub,key in [("qa_1","squad"),("qa_2","hotpotqa")]:
        ids=set((split/"ruler"/(sub+"_sources")/"D_cal.txt").read_text().splitlines())
        prefix = f"ruler/{sub}_sources/"
        input_data = json.loads((ruler/(key+".json")).read_bytes())
        if key=="squad":
            articles=[]
            for a in input_data["data"]:
                paragraphs=[]
                for p in a["paragraphs"]:
                    questions=[q for q in p["qas"] if prefix+q["id"] in ids]
                    if questions:
                        paragraphs.append(dict(p,qas=questions))
                if paragraphs:
                    articles.append(dict(a,paragraphs=paragraphs))
            view=dict(input_data,data=articles)
            ordered=[q["id"] for a in articles for p in a["paragraphs"] for q in p["qas"]]
        else:
            view=[r for r in input_data if prefix+r["_id"] in ids]
            ordered=[r["_id"] for r in view]
        assert {prefix+i for i in ordered}==ids
        (assets/(key+".json")).write_text(json.dumps(view))
        qa_ids[sub]=[prefix+i for i in ordered]
    sys.path[:0]=[str(scripts/"data"),str(scripts/"data/synthetic")]
    import constants
    from template import Templates
    tasks=yaml.safe_load((scripts/"synthetic.yaml").read_bytes())
    out = raw/"p0a-ruler-inputs"
    results=[]
    for cell in audit["checks"]["ruler_generation"]:
        if cell["split"]!="D_cal":
            continue
        name, length, seed = cell["subtask"], cell["length"], cell["seed"]
        custom=tasks[name]
        base=constants.TASKS[custom["task"]]
        template=Templates["meta-llama3"].format(task_template=base["template"])+base.get("answer_prefix","")
        where=out/str(length)
        entry=scripts/"data/synthetic"/(custom["task"]+".py")
        sys.argv=[str(entry),"--save_dir",str(where),"--save_name",name,"--subset","D_cal",
                  "--tokenizer_path",str(source/"llama"),"--tokenizer_type","hf","--max_seq_length",str(length),
                  "--num_samples","1","--random_seed",str(seed),"--template",template,
                  "--tokens_to_generate",str(base["tokens_to_generate"])]
        for k,v in custom["args"].items():sys.argv.extend(["--"+k,str(v)])
        runpy.run_path(str(entry),run_name="__main__")
        file=where/name/"D_cal.jsonl"
        row,=map(json.loads,file.read_text().splitlines())
        assert row["input"] and row["outputs"] and 0<row["length"]<=length
        digest=hashlib.sha256(row["input"].encode()).hexdigest()
        sample=f"ruler/{name}/{length}/D_cal/{row['index']}"
        parent=qa_ids[name][row["index"]] if name in qa_ids else sample
        results.append(dict(sample_id=sample,source_id=parent,model="llama",template="meta-llama3",
                            task=name,limit=length,actual_length=row["length"],seed=seed,
                            prompt_sha256=digest,file_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                            raw_file=str(file.relative_to(root))))
        print("RULER calibration fixture",name,length,flush=True)
    assert len(results)==26 and len({r["prompt_sha256"] for r in results})==26
    write_ids("ruler/initial_llama","D_cal",[r["sample_id"] for r in results])
    (run/"ruler_calibration.json").write_text(json.dumps(results,indent=2)+"\n")

if part=="wiki":
    wiki()
else:
    build_ruler()
