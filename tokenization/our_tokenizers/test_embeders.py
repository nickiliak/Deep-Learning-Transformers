from BPE.BPE_embedding import BPEEmbedder
from Canine.Canine_embedding import CanineEmbedder

text = "A quick brown fox jumps over the lazy dog."

bpe = BPEEmbedder("tokenization\\vocabularies\\bpe_tokenizer.json")
canine = CanineEmbedder()

v1 = bpe.generate_embedding(text)
v2 = canine.generate_embedding(text)

print(len(v1), len(v2))           # 768 768
print(f"BPE EMBEDDING {v1[:8]}")
print(f"CANINE EMBEDDING {v2[:8]}")