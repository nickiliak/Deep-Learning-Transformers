from tokenizers.BPE import CustomBPETokenizer

if __name__ == "__main__":
    tok = CustomBPETokenizer()
    #tok.load("my_bpe_tokenizer.json")
    tok.load("bpe_tokenizer.json")

    sample = "Eimai o kostis kai thelo na fame ena megalo souvlaki."
    ids = tok.encode(sample)
    print("Token IDs:", ids)

    decoded = tok.decode(ids)
    print("Decoded:", decoded)