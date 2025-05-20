import os
import gzip
import requests
import re
import multiprocessing
from pathlib import Path
from bs4 import BeautifulSoup

# Constants
COMMONCRAWL_BASE = "https://data.commoncrawl.org/"
SEGMENT_PATH = "crawl-data/CC-MAIN-2018-05/segments/1516084886237.6/wet/"
INDEX_URL = "https://data.commoncrawl.org/" + SEGMENT_PATH
OUTPUT_DIR = Path("commoncrawl_data_2018_05")
BUFFER_SIZE = 1024 * 1024  # 1MB
PARALLELISM = 8  # Number of parallel downloads

# Regex for cleaning content
WEB_PAT = re.compile(r"https?:[^ \n]* ")
WEB_REPL = "WEB "
WEB2_PAT = re.compile(r"https?:[^ \n]*\n")
WEB2_REPL = "WEB\n"


def fetch_file_list():
    """Fetches the list of all WET files in the specified segment."""
    response = requests.get(INDEX_URL)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    files = [
        a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".warc.wet.gz")
    ]
    
    if not files:
        raise ValueError("No WET files found in the specified segment.")
    
    return [SEGMENT_PATH + file for file in files]


def clean_content(raw_content: str) -> str:
    """Clean text by removing URLs and formatting issues."""
    par = raw_content.replace("</s>", ". ").replace("\t", " ")
    par = re.sub(WEB_PAT, WEB_REPL, par)
    par = re.sub(WEB2_PAT, WEB2_REPL, par)
    return par


def download_file(url: str, output_path: Path):
    """Download a file from a URL and save it locally."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=BUFFER_SIZE):
                f.write(chunk)

        print(f"Downloaded: {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")


def extract_gzip(file_path: Path, output_path: Path):
    """Extract a .gz file to a text file."""
    with gzip.open(file_path, "rt", encoding="utf-8") as gz_file, open(output_path, "w", encoding="utf-8") as out_file:
        for line in gz_file:
            out_file.write(line)
    print(f"Extracted: {output_path}")


def process_segment(segment_url: str, output_dir: Path):
    """Download, extract, and clean a WET file from Common Crawl."""
    segment_name = segment_url.split("/")[-1]
    gz_path = output_dir / segment_name
    txt_path = gz_path.with_suffix("")

    # Download
    download_file(COMMONCRAWL_BASE + segment_url, gz_path)

    # Extract
    extract_gzip(gz_path, txt_path)

    # Clean text
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    cleaned_text = clean_content(raw_content)
    cleaned_path = txt_path.with_suffix(".cleaned.txt")

    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"Cleaned: {cleaned_path}")


def download_commoncrawl_data(output_dir: Path):
    """Download and process all WET files from the specified segment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch the list of files
    segment_list = fetch_file_list()
    
    with multiprocessing.Pool(PARALLELISM) as pool:
        pool.map(lambda segment: process_segment(segment, output_dir), segment_list)


if __name__ == "__main__":
    download_commoncrawl_data(OUTPUT_DIR)
