import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import spacy.tokens
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL.Image import Image as PilImage
from transformers.tokenization_utils import PreTrainedTokenizer

from alfie.utils import auto_autocast, cached_nlp, compute_token_merge_indices

__all__ = [
    "CrossGlobalHeatMap",
    "CrossRawHeatMapCollection",
    "CrossWordHeatMap",
    "CrossParsedHeatMap",
    "CrossSyntacticHeatMapPair",
]

logger = logging.getLogger(__name__)


@dataclass
class CrossWordHeatMap:
    heatmap: torch.Tensor
    word: str

    @property
    def value(self):
        return self.heatmap

    def expand_as(
        self,
        image: PilImage,
        absolute: bool = False,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        im = rearrange(self.heatmap, "h w -> 1 1 h w")
        im = F.interpolate(
            im.float().detach(), size=(image.size[0], image.size[1]), mode="bicubic"
        )

        if not absolute:
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)

        if threshold:
            im = (im > threshold).float()

        # shape: (1, 1, h, w) -> (h, w)
        im = im.clone().detach().cpu().squeeze()

        return im


@dataclass
class CrossSyntacticHeatMapPair:
    head_heat_map: CrossWordHeatMap
    dep_heat_map: CrossWordHeatMap
    head_text: str
    dep_text: str
    relation: str


@dataclass
class CrossParsedHeatMap:
    word_heat_map: CrossWordHeatMap
    token: spacy.tokens.Token


@dataclass(frozen=True)
class CrossGlobalHeatMap:
    tokenizer: PreTrainedTokenizer
    prompt: str
    heat_maps: torch.Tensor

    @lru_cache(maxsize=50)
    def compute_word_heat_map(self, word: str) -> CrossWordHeatMap:
        merge_idxs = compute_token_merge_indices(self.tokenizer, self.prompt, word)
        return CrossWordHeatMap(self.heat_maps[merge_idxs].mean(0), word)

    def parsed_heat_maps(self) -> Iterable[CrossParsedHeatMap]:
        for token in cached_nlp(self.prompt):
            try:
                heat_map = self.compute_word_heat_map(token.text)
                yield CrossParsedHeatMap(heat_map, token)
            except ValueError as err:
                logger.warning(err)

    def dependency_relations(self) -> Iterable[CrossSyntacticHeatMapPair]:
        for token in cached_nlp(self.prompt):
            if token.dep_ != "ROOT":
                try:
                    dep_heat_map = self.compute_word_heat_map(token.text)
                    head_heat_map = self.compute_word_heat_map(token.head.text)

                    yield CrossSyntacticHeatMapPair(
                        head_heat_map,
                        dep_heat_map,
                        token.head.text,
                        token.text,
                        token.dep_,
                    )
                except ValueError as err:
                    logger.warning(err)


RawHeatMapKey = Tuple[int]  # layer


class CrossRawHeatMapCollection:
    def __init__(self):
        self.ids_to_heatmaps: Dict[RawHeatMapKey, torch.Tensor] = defaultdict(
            lambda: 0.0
        )
        self.ids_to_num_maps: Dict[RawHeatMapKey, int] = defaultdict(lambda: 0)
        self.device_type = None

    def update(self, layer_idx: int, heatmap: torch.Tensor):
        if self.device_type is None:
            self.device_type = heatmap.device.type
        with auto_autocast(device_type=self.device_type, dtype=torch.float32):
            key = layer_idx
            self.ids_to_heatmaps[key] = self.ids_to_heatmaps[key] + heatmap

    def factors(self) -> Set[int]:
        return set(key[0] for key in self.ids_to_heatmaps.keys())

    def layers(self) -> Set[int]:
        return set(key[1] for key in self.ids_to_heatmaps.keys())

    def heads(self) -> Set[int]:
        return set(key[2] for key in self.ids_to_heatmaps.keys())

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self) -> None:
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()
