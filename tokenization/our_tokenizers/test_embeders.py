from BPE.BPE_LSTM_embedding import BPELSTMEmbedder
from Canine.Canine_embedding import CanineEmbedder
from BPE.BPEpre import BPEPretrainedEmbedder
text = "A quick brown fox jumps over the lazy dog."

#bpe = BPELSTMEmbedder("tokenization\\vocabularies\\bpe_tokenizer.json")
bpe = BPEPretrainedEmbedder("roberta-base")
canine = CanineEmbedder()

v1 = bpe.generate_embedding(text)
v2 = canine.generate_embedding(text)

print(len(v1), len(v2))           # 768 768
print(f"BPE EMBEDDING {v1[:8]}")
print(f"CANINE EMBEDDING {v2[:8]}")