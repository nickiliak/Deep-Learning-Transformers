import json
class CustomBPETokenizer:
    def __init__(self):
        self.merges = {}         # (a,b) → new_token_id
        self.vocab_size = None
        self.num_merges = None

    # ------------------------------------------------------------
    # Utility 1: Pair frequency statistics
    # ------------------------------------------------------------
    @staticmethod
    def get_stats(ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    # ------------------------------------------------------------
    # Utility 2: Apply a merge operation on a token sequence
    # ------------------------------------------------------------
    @staticmethod
    def merge_tokens(ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if (
                i < len(ids) - 1
                and ids[i] == pair[0]
                and ids[i + 1] == pair[1]
            ):
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    # ------------------------------------------------------------
    # TRAINING METHOD
    # ------------------------------------------------------------
    def train(self, text, vocab_size=4000):
        """
        Train byte-pair encoding tokenizer on provided text.
        Base vocab: 0–255 for raw bytes.
        New tokens: 256+ for merges.
        """

        print("Initializing BPE training...")
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256

        tokens = list(text.encode("utf-8"))
        merges = {}

        for i in range(self.num_merges):
            stats = self.get_stats(tokens)
            if not stats:
                print("No more mergeable pairs.")
                break

            pair = max(stats, key=stats.get)
            new_id = 256 + i

            tokens = self.merge_tokens(tokens, pair, new_id)
            merges[pair] = new_id

            print(f"merge {i+1}/{self.num_merges}: {pair} → {new_id} ({stats[pair]} occurrences)")

        self.merges = merges
        print("Training complete.")

    # ------------------------------------------------------------
    # ENCODE - OPTIMIZED GREEDY ALGORITHM (Single Pass, O(n) time)
    # ------------------------------------------------------------
    def encode(self, text):
        """
        Encode text using greedy BPE algorithm.
        
        OPTIMIZATION: Single-pass greedy merge instead of scanning entire sequence
        each iteration. This is O(n) instead of O(n²) and matches how production
        tokenizers (SentencePiece, Tokenizers) work.
        
        Algorithm:
        1. Convert text to bytes
        2. Scan left-to-right one time
        3. At each position, check if (current, next) is a mergeable pair
        4. If yes, replace in-place; if no, move forward
        """
        ids = list(text.encode("utf-8"))
        
        # Single pass: greedily merge pairs
        i = 0
        while i < len(ids) - 1:
            pair = (ids[i], ids[i + 1])
            
            # If this pair has a merge rule, apply it
            if pair in self.merges:
                new_id = self.merges[pair]
                ids[i:i+2] = [new_id]  # Replace pair with merged token (in-place)
                # Don't increment i - check the newly merged token against next
            else:
                # No merge rule for this pair, move forward
                i += 1
        
        return ids

    # ------------------------------------------------------------
    # DECODE  (inverse BPE)
    # ------------------------------------------------------------
    def decode(self, ids):
        """
        Convert token IDs back to utf-8 text by fully expanding merges.
        """

        # Build reverse vocab: id → byte sequence
        vocab = self.build_vocab()

        # Expand each token id into bytes
        byte_sequence = b"".join(vocab[i] for i in ids)

        return byte_sequence.decode("utf-8", errors="replace")

    # ------------------------------------------------------------
    # Build vocabulary: token_id → byte sequence
    # ------------------------------------------------------------
    def build_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}

        # merge tokens
        # Reconstruct merges in order
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])

        for (a, b), idx in sorted_merges:
            vocab[idx] = vocab[a] + vocab[b]

        return vocab

    # ------------------------------------------------------------
    # SAVE + LOAD
    # ------------------------------------------------------------
    def save(self, path):
        obj = {
            "vocab_size": self.vocab_size,
            "num_merges": self.num_merges,
            "merges": {f"{a},{b}": idx for (a, b), idx in self.merges.items()},
        }
        with open(path, "w") as f:
            json.dump(obj, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        with open(path, "r") as f:
            obj = json.load(f)

        self.vocab_size = obj["vocab_size"]
        self.num_merges = obj["num_merges"]

        self.merges = {
            tuple(map(int, key.split(","))): val
            for key, val in obj["merges"].items()
        }
        print(f"Tokenizer loaded from {path}")