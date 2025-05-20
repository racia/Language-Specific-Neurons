import gzip
import logging
import re
import multiprocessing
import requests
from pathlib import Path
from typing import Dict, Iterable, NamedTuple, Type, Callable
import functools

BUFFER_SIZE = "32G"
SORT_PARALLEL = 8

KNOWN_VERSIONS = ["v1.0.0", "v1.0.beta", "v1.0.alpha"]

class NormalizedBitextPtr(NamedTuple):
    lang_pair: str
    line_no: int
    segment: str
    digest: str
    ptr_start: int
    ptr_end: int
    score: float

class Bitext(NamedTuple):
    lang_pair: str
    line_no: int
    score: float
    text: str

WEB_PAT = re.compile(r"https?:[^ \n]* ")
WEB_REPL = "WEB "

WEB2_PAT = re.compile(r"https?:[^ \n]*\n")
WEB2_REPL = "WEB\n"

def clean_content(raw_content: str) -> str:
    par = raw_content.replace("</s>", ". ").replace("\t", " ")
    par = re.sub(WEB_PAT, WEB_REPL, par)
    par = re.sub(WEB2_PAT, WEB2_REPL, par)
    return par

def get_typed_parser(cls: Type) -> Callable:
    types = cls.__annotations__.values()
    
    def parser(line: str) -> NamedTuple:
        parts = line.rstrip("\n").split("\t")
        assert len(parts) == len(types), f"Column mismatch: expected {cls.__annotations__}, got {parts}"
        return cls(*(t(p) for t, p in zip(types, parts)))
    
    return parser

def download_file(url: str, output_path: Path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        logging.error(f"Failed to download {url}, status code: {response.status_code}")

def extract_gzip(file_path: Path, output_path: Path):
    with gzip.open(file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
        f_out.write(f_in.read())

def download_and_extract(segment: str, outdir: Path):
    cc_base_url = "https://data.commoncrawl.org/"
    segment_url = cc_base_url + segment
    segment_path = outdir / Path(segment).name
    extracted_path = segment_path.with_suffix("")
    
    if not segment_path.exists():
        download_file(segment_url, segment_path)
    if not extracted_path.exists():
        extract_gzip(segment_path, extracted_path)
    
    return extracted_path

def process_segment(segment: str, outdir: Path):
    extracted_path = download_and_extract(segment, outdir)
    with open(extracted_path, "r", encoding="utf-8") as f:
        return {hash(line): line.strip() for line in f}

def dl(outdir: Path = Path("data"), version: str = KNOWN_VERSIONS[0], parallelism: int = 8):
    assert version in KNOWN_VERSIONS, f"Unknown version {version}, choose from {KNOWN_VERSIONS}"
    metadata_url = f"https://dl.fbaipublicfiles.com/laser/CCMatrix/{version}/list.txt"
    response = requests.get(metadata_url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch metadata list from {metadata_url}")
        return
    
    file_list = response.text.strip().split("\n")
    outdir.mkdir(exist_ok=True, parents=True)
    outdir = outdir / version / "raw"
    outdir.mkdir(exist_ok=True, parents=True)
    
    with multiprocessing.Pool(parallelism) as pool:
        pool.map(functools.partial(process_segment, outdir=outdir), file_list)

if __name__ == "__main__":
    dl()
