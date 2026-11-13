"""Resolve the pinned RULER essay list, preserve raw bytes and the official conversion order."""
import concurrent.futures
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import html2text
from bs4 import BeautifulSoup

run = Path(__file__).resolve().parent
ruler = Path("data/raw/sources/ruler")
rev = "021385d68d3202e37893e9d3cd29011c569abe30"
repo = ruler / "essay_repo.tar.gz"
raw = ruler / "essay_inputs"
raw.mkdir(exist_ok=True)
def download(url, path):
    tmp = path.with_suffix(path.suffix+".part")
    subprocess.run(["curl", "-fsSL", "--connect-timeout", "12", "--max-time","75",
                    "--retry", "2", "-o", str(tmp), url],check=True)
    tmp.replace(path)
if not repo.exists():
    download("https://codeload.github.com/gkamradt/LLMTest_NeedleInAHaystack/tar.gz/"+rev,repo)
with tarfile.open(next(ruler.glob("db43*.tar.gz"))) as tar:
    member, = [p for p in tar.getnames() if p.endswith("/PaulGrahamEssays_URLs.txt")]
    urls = tar.extractfile(member).read().decode().splitlines()
html = [u for u in urls if u.endswith(".html")]
def fetch_html(u):
    target = raw / u.rsplit("/",1)[1]
    if not target.exists():
        download(u.replace("http:", "https:"),target)
    body = target.read_bytes()
    if b"<html" not in body.lower() or len(body)<500:
        raise ValueError(u)
with concurrent.futures.ThreadPoolExecutor(8) as ex:
    list(ex.map(fetch_html,html))
convert = html2text.HTML2Text()
convert.ignore_images = convert.ignore_tables = convert.escape_all = True
convert.reference_links = convert.mark_code = False
source_hashes = {}
texts = []
with tarfile.open(repo) as tar:
    for u in sorted(set(urls)-set(html),key=lambda s:s.rsplit("/",1)[1]):
        name = u.rsplit("/",1)[1]
        member, = [p for p in tar.getnames() if p.endswith("/needlehaystack/PaulGrahamEssays/"+name)]
        body = tar.extractfile(member).read()
        texts.append(body.decode())
        source_hashes[u] = hashlib.sha256(body).hexdigest()
for u in sorted(html,key=lambda s:s.rsplit("/",1)[1]):
    body = (raw/u.rsplit("/",1)[1]).read_bytes()
    tag = BeautifulSoup(body.decode("unicode_escape","replace"),"html.parser").find("font")
    if tag is None: raise ValueError(u)
    texts.append(convert.handle(str(tag)))
    source_hashes[u] = hashlib.sha256(body).hexdigest()
out = ruler/"PaulGrahamEssays.json"
out.write_bytes(json.dumps({"text":"".join(texts)}).encode())
(run/"essay_sources.json").write_text(json.dumps(dict(repo_revision=rev, source_count=len(urls),
    source_sha256=source_hashes, corpus_sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2)+"\n")
print("verified corpus",len(urls), out.stat().st_size)
