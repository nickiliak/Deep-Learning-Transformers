from BPE import CustomBPETokenizer
from Canine import CanineTokenizer
from ByT5 import ByT5Tokenizer


def main():
    sample = "O gazis einai gay kai tou aresoun ta paidia."

    # ---- BPE ----
    print("=== Custom BPE ===")
    bpe = CustomBPETokenizer()
    bpe.load("../vocabularies/bpe_tokenizer.json")  # path to your saved BPE model
    bpe_ids = bpe.encode(sample)
    print("BPE ids:     ", bpe_ids)
    print("BPE decoded: ", bpe.decode(bpe_ids))
    print("BPE vocab size:", len(bpe_ids))
    print()

    # ---- CANINE ----
    print("=== CANINE (google/canine-s) ===")
    canine = CanineTokenizer()
    canine_ids = canine.encode(sample)
    print("CANINE ids (len):", len(canine_ids))
    print("CANINE ids (head):", canine_ids[:20])
    print("CANINE decoded:", canine.decode(canine_ids))
    print()

    # ---- ByT5 ----
    print("=== ByT5 (google/byt5-small) ===")
    byt5 = ByT5Tokenizer()
    byt5_ids = byt5.encode(sample)
    print("ByT5 ids (len):", len(byt5_ids))
    print("ByT5 ids (head):", byt5_ids[:20])
    print("ByT5 decoded:", byt5.decode(byt5_ids))
    print()


if __name__ == "__main__":
    main()