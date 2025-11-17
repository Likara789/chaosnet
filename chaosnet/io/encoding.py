# chaosnet/io/encoding.py

import torch


class CharEncoder:
    """
    Super simple character-level encoder/decoder.
    This is just here so you have some IO machinery to build on later.
    """

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode_indices(self, text: str):
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode_indices(self, idx: torch.Tensor):
        return "".join(self.itos[int(i)] for i in idx)

    def one_hot(self, text: str):
        idx = self.encode_indices(text)
        x = torch.zeros(len(idx), self.vocab_size)
        x[torch.arange(len(idx)), idx] = 1.0
        return x
