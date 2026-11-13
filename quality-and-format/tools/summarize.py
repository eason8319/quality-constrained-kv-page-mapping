"""Generate the geometry-only E0 byte model from the same layout used by the packer."""
import json
from dataclasses import asdict
from pathlib import Path
from pack.layout import grid, layout

def byte_model():
    rows=[]
    for length in (32,64,128):
        for c in grid(length):
            for burst in (32,64):
                rows.append(dict(page_length=length,head_dim=128,burst=burst,**asdict(c),
                                 **layout(c,length,128,burst)))
    return dict(protocol_version="affine_min",rows=rows)

if __name__=="__main__":
    dest=Path("artifacts/byte_model.json")
    dest.parent.mkdir(exist_ok=True)
    tmp=dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(byte_model(),indent=2)+"\n")
    tmp.replace(dest)
