from dataclasses import dataclass
from typing import Dict, Optional, TypedDict

import torch
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput


class AlfieHeatmaps(TypedDict):
    cross_heatmaps: Dict[str, torch.Tensor]
    fg_nouns: Dict[str, torch.Tensor]
    self_heatmaps_fg_nouns: Dict[str, torch.Tensor]
    cross_heatmap_fg: Dict[str, torch.Tensor]
    ff_heatmap: Dict[str, torch.Tensor]


@dataclass
class AlfieImagePipelineOutput(ImagePipelineOutput):
    heatmaps: Optional[AlfieHeatmaps] = None
