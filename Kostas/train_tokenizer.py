from datasets import load_dataset
import os
from tokenizers.BPE import CustomBPETokenizer
# ------------------------------------------------------------
# STEP 1 — Build corpus.txt from C4
# ------------------------------------------------------------

def dump_to_corpus(n_docs=50_000, out="corpus.txt"):
    ds = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)

    with open(out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i >= n_docs:
                break
            text = ex["text"].replace("\n", " ")
            f.write(text + "\n")

    print(f"Wrote {n_docs} documents to {out}")


dump_to_corpus()

'''
def dump_to_corpus_c4_beir(n_c4_docs=500_000, out="corpus.txt"):
    from datasets import load_dataset

    c4 = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
    trec = load_dataset("BeIR/trec-news", "corpus", split="train")

    with open(out, "w", encoding="utf-8") as f:
        # C4 part
        for i, ex in enumerate(c4):
            if i >= n_c4_docs:
                break
            f.write(ex["text"].replace("\n", " ") + "\n")

        # BeIR part
        for ex in trec:
            f.write(ex["text"].replace("\n", " ") + "\n")
    '''
# ------------------------------------------------------------
# STEP 2 — Load corpus
# ------------------------------------------------------------
if __name__ == "__main__":
    #dump_to_corpus_c4_beir()
    dump_to_corpus(n_docs=50_000, out="corpus.txt") #sanity check
    print(os.path.getsize("corpus.txt") / 1e6, "MB")
    with open("corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CustomBPETokenizer()
    tokenizer.train(text, vocab_size=2000) #small vocab for testing
    tokenizer.save("bpe_tokenizer.json")

    vocab = tokenizer.build_vocab()
    with open("corpus.txt", "w", encoding="utf-8") as f:
        for idx in sorted(vocab):
            f.write(f"{idx}\t{vocab[idx]}\n")

    print("Done!")