import os
import sys
import time
import urllib.request

DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "wavefake_samples")
DEST_FILE = os.path.join(DEST_DIR, "partition0.parquet")
URL = "https://huggingface.co/datasets/ajaykarthick/wavefake-audio/resolve/main/data/partition0-00000-of-00001.parquet"

def download_partition():
    os.makedirs(DEST_DIR, exist_ok=True)
    if os.path.exists(DEST_FILE):
        size = os.path.getsize(DEST_FILE)
        if size > 200 * 1024 * 1024:
            print(f"Partition already downloaded ({size / (1024*1024):.2f} MB): {DEST_FILE}")
            return DEST_FILE

    print(f"Downloading WaveFake Partition 0 (222 MB) from Hugging Face...")
    print(f"Target: {DEST_FILE}")
    
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 TrueToneSIH/1.0"})
    with urllib.request.urlopen(req) as resp, open(DEST_FILE, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        t0 = time.time()
        last_print = t0

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 2.0 or downloaded == total_size:
                pct = (downloaded / total_size) * 100 if total_size else 0
                speed = (downloaded / (now - t0)) / (1024 * 1024) if (now - t0) > 0 else 0
                print(f"Progress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", flush=True)
                last_print = now

    print(f"\nDownload completed successfully: {DEST_FILE} ({os.path.getsize(DEST_FILE) / (1024*1024):.2f} MB)")
    return DEST_FILE

if __name__ == "__main__":
    download_partition()
