import urllib.request
import os
import gzip
import shutil
import sys

def download_simplified_nq():
    os.makedirs("data", exist_ok=True)

    files = {
        "train": "https://storage.googleapis.com/natural_questions/v1.0/train/nq-train-00.jsonl.gz",
        "dev":   "https://storage.googleapis.com/natural_questions/v1.0/dev/nq-dev-00.jsonl.gz"
    }

    for split, url in files.items():
        gz_path = os.path.join("data", f"{split}.jsonl.gz")
        json_path = os.path.join("data", f"{split}.jsonl")

        if not os.path.exists(gz_path):
            print(f"Downloading {split} split...")
            try:
                urllib.request.urlretrieve(url, gz_path)
            except Exception as e:
                print(f"Failed to download {url}: {e}", file=sys.stderr)
                continue
        else:
            print(f"{split} split already downloaded.")

        if not os.path.exists(json_path):
            print(f"Extracting {split} split...")
            try:
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(json_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                print(f"Extraction failed for {gz_path}: {e}", file=sys.stderr)
                # keep the .gz for debugging / retry
                continue

            # remove the compressed file after successful extraction
            try:
                os.remove(gz_path)
                print(f"Removed compressed file: {gz_path}")
            except OSError as e:
                print(f"Could not remove {gz_path}: {e}", file=sys.stderr)
        else:
            print(f"{split} split already extracted.")

if __name__ == "__main__":
    download_simplified_nq()