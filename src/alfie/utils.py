import random
from functools import lru_cache
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import spacy
import torch

T = TypeVar("T")


__all__ = [
    "set_seed",
    "compute_token_merge_indices",
    "plot_mask_heat_map",
    "cached_nlp",
    "auto_device",
    "auto_autocast",
]


T = TypeVar("T")


def auto_device(obj: T = torch.device("cpu")) -> T:
    if isinstance(obj, torch.device):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        return obj.to("cuda")

    return obj


def auto_autocast(*args, **kwargs):
    if not torch.cuda.is_available():
        kwargs["enabled"] = False

    return torch.autocast(*args, **kwargs)


def plot_mask_heat_map(
    im: PIL.Image.Image, heat_map: torch.Tensor, threshold: float = 0.4
):
    im = torch.from_numpy(np.array(im)).float() / 255
    mask = (heat_map.squeeze() > threshold).float()
    im = im * mask.unsqueeze(-1)
    plt.imshow(im)


def normalize_masks(x: torch.Tensor):
    return (x - x.min()) / (x.max() - x.min())


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device=auto_device())
    gen.manual_seed(seed)

    return gen


def compute_token_merge_indices(tokenizer, prompt: str, word: str):
    merge_idxs = []
    tokens = tokenizer.tokenize(prompt.lower())
    tokens = [
        x.replace("</w>", "") for x in tokens
    ]  # New tokenizer uses wordpiece markers.

    search_tokens = [
        x.replace("</w>", "") for x in tokenizer.tokenize(word.lower())
    ]  # New tokenizer uses wordpiece markers.
    start_indices = [
        x
        for x in range(len(tokens))
        if tokens[x : x + len(search_tokens)] == search_tokens
    ]

    for indice in start_indices:
        merge_idxs += [i + indice for i in range(0, len(search_tokens))]

    if not merge_idxs:
        prompt = prompt.replace('"', "")
        tokens = [
            x.replace("</w>", "") for x in tokenizer.tokenize(prompt.lower())
        ]  # New tokenizer uses wordpiece markers.
        start_indices = [
            x
            for x in range(len(tokens))
            if tokens[x : x + len(search_tokens)] == search_tokens
        ]
        for indice in start_indices:
            merge_idxs += [i + indice for i in range(0, len(search_tokens))]
        if not merge_idxs:
            raise ValueError(f"Search word {word} not found in prompt!")

    return [x for x in merge_idxs]


nlp = None


@lru_cache(maxsize=100000)
def cached_nlp(prompt: str, type="en_core_web_md"):
    global nlp

    if nlp is None:
        try:
            nlp = spacy.load(type)
        except OSError:
            import os

            os.system(f"python -m spacy download {type}")
            nlp = spacy.load(type)

    return nlp(prompt)
