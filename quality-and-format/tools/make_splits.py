"""Build P0a ID lists from pinned offline inputs. Run from quality-and-format/."""
import argparse
import csv
import hashlib
import json
import re
import tarfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml


SPLITS = ("D_cal", "D_sel", "D_test")
RAW = Path("data/raw/sources")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def parquet(key, pattern):
    path, = sorted((RAW / key).glob(pattern))
    return pq.read_table(path).to_pylist()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def revision_member(key, rev, suffix):
    with tarfile.open(RAW / key / (rev + ".tar.gz")) as archive:
        name, = [name for name in archive.getnames() if name.endswith("/" + suffix)]
        return archive.extractfile(name).read()


def quotas(sizes, count):
    total = sum(sizes.values())
    base = {k: n * count // total for k, n in sizes.items()}
    order = sorted(sizes, key=lambda k: (-(sizes[k] * count % total), k))
    for k in order[:count - sum(base.values())]:
        base[k] += 1
    return base


class Inventory:
    def __init__(self, seed):
        self.seed = seed
        self.rows = {}
        self.checks = {}

    def add(self, task, source_id, text, question=None):
        sample = f"{task}/{source_id}"
        require(sample not in self.rows and normalized(text), f"Invalid sample {sample}")
        self.rows[sample] = dict(task=task, sample_id=sample, raw_sha256=sha(text),
                                 group_sha256=sha(normalized(text)), split="reserve",
                                 question_sha256=sha(normalized(question)) if question else "")
        return self.rows[sample]

    def ordered(self, tag, values):
        return sorted(values, key=lambda v: sha(f"{self.seed}/{tag}/{v}"))

    def groups(self, tag, groups, fractions, forced=None):
        # Unit is the independent source group. Preserve forced cross-task matches.
        forced = forced or {}
        keys = self.ordered(tag, groups)
        limits = {s: round(fractions[s] * len(keys)) for s in SPLITS[:2]}
        limits["D_test"] = len(keys) - sum(limits.values())
        counts = Counter(forced.values())
        for key in keys:
            split = forced.get(key)
            if split is None:
                split = next((s for s in SPLITS if counts[s] < limits[s]), "D_test")
                counts[split] += 1
            for row in groups[key]:
                row["split"] = split

    def math(self, task, config, train, test, fewshot=()):
        dev = config["splits"]["development"]
        require(len(train) == dev["expected_pool_n"] and
                len(test) == config["splits"]["test"]["expected_n"], f"{task} source counts")
        blocked = set(fewshot)
        for source_id, question in test:
            row = self.add(task, source_id, question)
            row["split"] = "D_test"
            blocked.add(row["group_sha256"])
        candidates = defaultdict(dict)
        seen = set()
        for subject, source_id, question in train:
            row = self.add(task, source_id, question)
            key = row["group_sha256"]
            if key in blocked or key in seen:
                row["split"] = "excluded_overlap" if key in blocked else "excluded_duplicate"
                continue
            candidates[subject][key] = row
            seen.add(key)
        sizes = {k: len(v) for k, v in candidates.items()}
        cal = quotas(sizes, dev["cal_n"])
        sel = quotas(sizes, dev["sel_n"])
        for subject, group in candidates.items():
            order = self.ordered(f"{task}/{subject}", group)
            require(len(order) >= cal[subject] + sel[subject], "Insufficient training rows")
            for key in order[:cal[subject]]:
                group[key]["split"] = "D_cal"
            for key in order[cal[subject]:cal[subject]+sel[subject]]:
                group[key]["split"] = "D_sel"
        self.checks[task + "_subject_quotas"] = {"D_cal": cal, "D_sel": sel}

    def build(self, config):
        gsm = config["gsm8k"]
        spec = yaml.safe_load(revision_member("harness", gsm["harness_revision"],
                                              "lm_eval/tasks/gsm8k/gsm8k-cot.yaml"))
        shots = [sha(normalized(r["question"])) for r in spec["fewshot_config"]["samples"]]
        require(len(set(shots)) == gsm["num_fewshot"] == 8, "Fixed GSM8K examples")
        self.checks["gsm8k_fewshot_question_sha256"] = shots
        train = [("main", f"train/{i}", r["question"]) for i, r in
                 enumerate(parquet("gsm8k", "main/train*.parquet"))]
        test = [(f"test/{i}", r["question"]) for i, r in enumerate(parquet("gsm8k", "main/test*.parquet"))]
        self.math("gsm8k", gsm, train, test, shots)
        math = config["math500"]
        train = [(subject, f"train/{subject}/{i}", r["problem"])
                 for subject in math["splits"]["development"]["configs"]
                 for i, r in enumerate(parquet("mathtrain", subject + "/train*.parquet"))]
        test = [json.loads(line) for line in (RAW / "math500/test.jsonl").read_text().splitlines()]
        self.math("math500", math, train, [(r["unique_id"], r["problem"]) for r in test])

        aime = config["aime2024"]
        groups = defaultdict(list)
        keys = {"I": [204,25,809,116,104,294,540,197,480,113,371,385,110,104,721],
                "II": [73,236,45,33,80,55,699,127,902,468,601,23,321,211,315]}
        matched = set()
        for r in parquet("aime", "data/train*.parquet"):
            paper, num = re.search(r"2024_AIME_(II?)_Problems/Problem_(\d+)", r["url"]).groups()
            require(1 <= int(num) <= 15 and int(r["answer"]) == keys[paper][int(num)-1],
                    f"Answer mismatch {paper}/{num}")
            matched.add((paper, num))
            row = self.add("aime2024", r["id"], r["problem"])
            groups[row["group_sha256"]].append(row)
        n = aime["expected_n"]
        require(len(groups) == len(matched) == n, "AIME item coverage")
        self.groups("aime2024", groups, {s: aime["splits"][s[2:]+"_n"]/n for s in SPLITS})
        self.checks["aime_answer_keys_matched"] = n

        lb = config["longbench"]
        question_split = {}
        with zipfile.ZipFile(RAW / "longbench" / lb["data_archive"]) as archive:
            for sub in lb["subtasks"]:
                groups = defaultdict(list)
                name, = [p for p in archive.namelist() if p.split("/")[-1] == sub+".jsonl"]
                for line in archive.read(name).splitlines():
                    r = json.loads(line)
                    row = self.add("longbench/"+sub, r["_id"], r["context"], r["input"])
                    groups[row["group_sha256"]].append(row)
                self.groups("longbench/"+sub, groups, fractions(lb))
                if sub == "hotpotqa":
                    for rows in groups.values():
                        for r in rows:
                            q, s = r["question_sha256"], r["split"]
                            require(question_split.setdefault(q, s) == s, "LongBench question overlap")
        self.ruler(config["ruler"], question_split, lb["natural_evidence"]["source_file_sha256"])

        articles = []
        for r in parquet("wikitext", "wikitext-2-raw-v1/validation*.parquet"):
            line = r["text"]
            if re.fullmatch(r"= (?![= ])[^=].*? =", line.strip()) and not line.strip().endswith("= ="):
                articles.append([])
            if articles:
                articles[-1].append(line)
            else:
                require(not line.strip(), "Text before WikiText article header")
        groups = defaultdict(list)
        for i, lines in enumerate(articles):
            row = self.add("wikitext2", f"validation/article/{i}", "\n".join(lines))
            groups[row["group_sha256"]].append(row)
        require(bool(groups), "Empty WikiText article pool")
        self.groups("wikitext2", groups, fractions(config["wikitext2"]))

    def ruler(self, config, longbench, expected_hash):
        official = yaml.safe_load(revision_member("ruler", config["revision"], "scripts/synthetic.yaml"))
        require(set(config["subtasks"]) == set(official), "RULER tasks changed")
        self.checks["ruler_generation"] = [
            dict(subtask=t, length=n, split=s, seed=int(sha(f"{self.seed}/{t}/{n}/{s}")[:8], 16),
                 official_parameters=official[t])
            for t in config["subtasks"] for n in config["lengths"] for s in SPLITS]
        squad = json.loads((RAW / "ruler/squad.json").read_bytes())
        groups = defaultdict(list)
        for article in squad["data"]:
            for p in article["paragraphs"]:
                for q in p["qas"]:
                    if q["is_impossible"]:
                        continue
                    row = self.add("ruler/qa_1_sources", q["id"], p["context"], q["question"])
                    groups[row["group_sha256"]].append(row)
        # Group both repeated questions and shared source paragraphs.
        self.qa_groups("ruler/qa_1_sources", groups, {})

        data = (RAW / "ruler/hotpotqa.json").read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected_hash, "HotpotQA digest mismatch")
        groups = defaultdict(list)
        for item in json.loads(data):
            context = json.dumps(item["context"], ensure_ascii=False, separators=(",", ":"))
            row = self.add("ruler/qa_2_sources", item["_id"], context, item["question"])
            groups[row["group_sha256"]].append(row)
        self.qa_groups("ruler/qa_2_sources", groups, longbench)

    def qa_groups(self, task, groups, known):
        parents = {k:k for k in groups}
        def root(k):
            while parents[k] != k:
                parents[k] = parents[parents[k]]
                k = parents[k]
            return k
        questions = {}
        for k, rows in groups.items():
            for r in rows:
                previous = questions.setdefault(r["question_sha256"], k)
                parents[root(k)] = root(previous)
        joined = defaultdict(list)
        forced = {}
        hits = 0
        for k, rows in groups.items():
            key = root(k)
            joined[key].extend(rows)
            for r in rows:
                if r["question_sha256"] in known:
                    s = known[r["question_sha256"]]
                    require(forced.setdefault(key, s) == s, "Conflicting cross-task grouping")
                    hits += 1
        self.groups(task, joined, {"D_cal": .2, "D_sel": .2}, forced)
        self.checks[task] = {"independent_source_groups": len(joined), "longbench_matched_rows": hits}

    def save(self, output, audit):
        lists = defaultdict(list)
        seen = {}
        counts = defaultdict(Counter)
        for row in self.rows.values():
            task, split = row["task"], row["split"]
            counts[task][split] += 1
            if split not in SPLITS:
                continue
            for key in (row["group_sha256"], row["question_sha256"]):
                if key:
                    require(seen.setdefault(key, split) == split, "Source content spans splits: " + key)
            lists[(task, split)].append(row["sample_id"])
        summary = {"seed": self.seed, "counts": counts, "checks": self.checks, "files": {}}
        outputs = {}
        for (task, split), ids in sorted(lists.items()):
            path = Path(task) / (split + ".txt")
            data = "\n".join(sorted(ids)) + "\n"
            outputs[output / path] = data
            summary["files"][path.as_posix()] = sha(data)
        audit.mkdir(parents=True, exist_ok=True)
        for path, data in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data.encode())
        with (audit/"source_index.tsv").open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=next(iter(self.rows.values())).keys(), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows.values())
        (audit/"split_audit.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf8")
        print(json.dumps(counts, indent=2))


def fractions(config):
    return {s: config["splits"]["fractions"][s[2:]] for s in SPLITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/data_manifest.yaml"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/splits"))
    parser.add_argument("--audit-dir", type=Path, default=Path("data/raw/p0a-audit"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.manifest.read_text(encoding="utf8"))
    require(args.seed == cfg["split_seed"], "Seed must match data_manifest.yaml")
    inventory = Inventory(args.seed)
    inventory.build({d["task"]: d for d in cfg["datasets"]})
    inventory.save(args.output, args.audit_dir)


if __name__ == "__main__":
    main()
