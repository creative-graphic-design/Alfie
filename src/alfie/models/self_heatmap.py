import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL.Image import Image as PilImage

from alfie.utils import auto_autocast

__all__ = [
    "SelfGlobalHeatMap",
    "SelfRawHeatMapCollection",
    "SelfPixelHeatMap",
    "SelfParsedHeatMap",
    "SelfSyntacticHeatMapPair",
]


@dataclass
class SelfPixelHeatMap(object):
    heatmap: torch.Tensor

    @property
    def value(self):
        return self.heatmap

    def expand_as(self, image: PilImage) -> torch.Tensor:
        im = rearrange(self.heatmap, "h w -> 1 1 h w")
        im = F.interpolate(
            im.float().detach(), size=(image.size[0], image.size[1]), mode="bicubic"
        )
        im = im[0, 0]
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
        im = im.clone().detach().cpu().squeeze()

        return im


@dataclass
class SelfSyntacticHeatMapPair:
    head_heat_map: SelfPixelHeatMap
    dep_heat_map: SelfPixelHeatMap
    head_text: str
    dep_text: str
    relation: str


@dataclass
class SelfParsedHeatMap:
    word_heat_map: SelfPixelHeatMap


@dataclass
class SelfGlobalHeatMap(object):
    def __init__(self, heat_maps: torch.Tensor, latent_hw: int):
        self.heat_maps = heat_maps

        # The dimensions of the latent image on which the heatmap is generated
        self.latent_h = self.latent_w = int(math.sqrt(latent_hw))

        # The pixels for which the heatmap is generated (It can be condensed form and thus smaller compared to self.latent_h and latent_w)
        self.inner_latent_h = self.inner_latent_w = int(math.sqrt(heat_maps.shape[0]))

        # Scale Factor
        self.scale_factor = self.latent_h // self.inner_latent_h

    def compute_guided_heat_map(self, guide_heatmap: torch.tensor):
        """
        For each pixel in the latent image we have one heatmap. Now, with a guiding heatmap
        we can merge all these pixel heatmaps with a weighted average according to the weights
        given to each pixel in the guiding heatmap.

        guide_heatmap: A guiding heatmap of the dimension of the latent image. It should be a 2D torch.tensor
        """

        # convert the latent 2d image from height.width x height x width to 1 x height.weight x height x width
        # i.e. we add the batch dim
        heat_maps2d = rearrange(self.heat_maps, "c h w -> 1 c h w")
        heat_maps2d = heat_maps2d.detach().clone()

        # weight of the convolution layer that performs attention diffusion (making a copy to prevent changing the heatmap)
        conv_weight: torch.Tensor = rearrange(guide_heatmap, "h w -> 1 (h w) 1 1")
        conv_weight = conv_weight.detach().clone()

        # For getting weighted average after 1x1 Kernel convolution below
        conv_weight /= conv_weight.sum(dim=1, keepdims=True)
        guided_heatmap = F.conv2d(heat_maps2d, conv_weight)[0, 0]

        return SelfPixelHeatMap(guided_heatmap.cpu() * guide_heatmap.cpu())


@dataclass
class SelfRawHeatMapCollection:
    ids_to_heatmaps: Dict[int, torch.Tensor] = field(default_factory=dict)
    ids_to_num_maps: Dict[int, int] = field(default_factory=dict)
    device_type: Optional[str] = None

    def update(self, layer_idx: int, heatmap: torch.Tensor):
        if self.device_type is None:
            self.device_type = heatmap.device.type

        with auto_autocast(device_type=self.device_type, dtype=torch.float32):
            key = layer_idx

            # Instead of simple addition can we do something better ???
            if key not in self.ids_to_heatmaps:
                self.ids_to_heatmaps[key] = heatmap
            else:
                self.ids_to_heatmaps[key] = self.ids_to_heatmaps[key] + heatmap

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self):
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()
