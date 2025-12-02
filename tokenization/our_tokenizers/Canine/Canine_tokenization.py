# tokenizers/CanineTokenizer.py

from typing import List, Union
from transformers import AutoTokenizer


class CanineTokenizer:
    """
    Simple wrapper around the pretrained CANINE tokenizer.

    - Uses 'google/canine-s' by default.
    - Interface: encode(text) -> List[int], decode(List[int]) -> str
    """

    def __init__(self, pretrained_name: str = "google/canine-s") -> None:
        self.pretrained_name = pretrained_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a single string into CANINE token IDs.
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return encoded["input_ids"]

    def encode_batch(
        self, texts: List[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        """
        Encode a batch of texts.
        """
        encoded = self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return encoded["input_ids"]

    def decode(self, ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens) \
            if isinstance(ids[0], int) else [
                self.tokenizer.decode(seq, skip_special_tokens=skip_special_tokens)
                for seq in ids
            ]