"""Atomic experiment outputs; a failed write cannot masquerade as a case."""
import gzip
import json
from pathlib import Path

def _commit(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_bytes(body)
    temporary.replace(path)

def save_json(path,value):
    _commit(path,json.dumps(value,indent=2,allow_nan=False).encode('utf-8'))

def save_gzip(path,value):
    _commit(path,gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode('utf-8'),mtime=0))
