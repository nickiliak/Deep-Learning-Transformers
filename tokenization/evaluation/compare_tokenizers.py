import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from our_tokenizers.BPE import CustomBPETokenizer

import csv
from transformers import AutoTokenizer

# -----------------------------
# Load tokenizers
# -----------------------------

# 1. Load your custom BPE class + the vocabulary you trained
bpe = CustomBPETokenizer()
bpe.load("../vocabularies/bpe_tokenizer.json")  # <--- your saved merges + vocab

# 2. Load CANINE (pretrained, fixed character-level tokenizer)
canine_tok = AutoTokenizer.from_pretrained("google/canine-s")

# 3. Load ByT5 (pretrained, fixed byte-level tokenizer)
byt5_tok = AutoTokenizer.from_pretrained("google/byt5-small")


# -----------------------------
# Statistics accumulators
# -----------------------------

stats = {
    "num_lines": 0,
    "bpe_total": 0,
    "canine_total": 0,
    "byt5_total": 0,

    "bpe_max": 0,
    "canine_max": 0,
    "byt5_max": 0,
}


# -----------------------------
# CSV output
# -----------------------------

with open("tokenizer_comparison.csv", "w", newline="", encoding="utf-8") as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(["line_id", "bpe_len", "canine_len", "byt5_len"])

    # -------------------------
    # Stream over corpus
    # -------------------------
    with open("corpus.txt", "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):

            line = line.strip()
            if not line:
                continue

            # --- Custom BPE ---
            bpe_ids = bpe.encode(line)
            bpe_len = len(bpe_ids)

            # --- CANINE ---
            canine_ids = canine_tok(line, add_special_tokens=True)["input_ids"]
            canine_len = len(canine_ids)

            # --- ByT5 ---
            byt5_ids = byt5_tok(line, add_special_tokens=True)["input_ids"]
            byt5_len = len(byt5_ids)

            # --- Update stats ---
            stats["num_lines"] += 1
            stats["bpe_total"] += bpe_len
            stats["canine_total"] += canine_len
            stats["byt5_total"] += byt5_len

            stats["bpe_max"] = max(stats["bpe_max"], bpe_len)
            stats["canine_max"] = max(stats["canine_max"], canine_len)
            stats["byt5_max"] = max(stats["byt5_max"], byt5_len)

            # --- Save per-line stats ---
            writer.writerow([line_id, bpe_len, canine_len, byt5_len])


# -----------------------------
# Final summary
# -----------------------------

print("\n=== Tokenization Statistics ===")
print(f"Processed lines: {stats['num_lines']}")

print("\nAverage token lengths:")
print(f"BPE:    {stats['bpe_total'] / stats['num_lines']:.2f}")
print(f"CANINE: {stats['canine_total'] / stats['num_lines']:.2f}")
print(f"ByT5:   {stats['byt5_total'] / stats['num_lines']:.2f}")

print("\nMaximum token lengths:")
print(f"BPE:    {stats['bpe_max']}")
print(f"CANINE: {stats['canine_max']}")
print(f"ByT5:   {stats['byt5_max']}")

print("\nSaved detailed results to tokenizer_comparison.csv")