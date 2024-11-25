from typing import List

import pytest
import torch
from PIL.Image import Image as PilImage

from alfie.pipelines import AlfiePixArtSigmaPipeline
from alfie.pipelines.pixart_alpha import CutoutModel


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def torch_dtype() -> torch.dtype:
    return torch.float16


@pytest.fixture
def disable_centering() -> bool:
    return False


@pytest.fixture
def k_alpha_intensity() -> float:
    return 0.5


@pytest.fixture
def negative_prompts() -> List[str]:
    return ["Blurry, shadow, low-resolution, low-quality"]


@pytest.fixture
def steps() -> int:
    return 30


@pytest.fixture
def sure_fg_threshold() -> float:
    return 0.8


@pytest.fixture
def maybe_fg_threshold() -> float:
    return 0.3


@pytest.fixture
def maybe_bg_threshold() -> float:
    return 0.1


@pytest.fixture
def num_images() -> int:
    return 3


@pytest.fixture
def seed() -> int:
    return 2024


@pytest.fixture
def vit_matte_model_id() -> str:
    return "hustvl/vitmatte-base-composition-1k"


@pytest.fixture
def pipeline_id() -> str:
    return "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"


@pytest.mark.parametrize(
    argnames="fg_prompt",
    argvalues=(
        "A photo of a cat with a hat",
        "A large, colorful tree made of money, with lots of yellow and white coins hanging from its branches",
    ),
)
@pytest.mark.parametrize(
    argnames="image_size, denoiser_id",
    argvalues=(
        (256, "PixArt-alpha/PixArt-Sigma-XL-2-256x256"),
        (512, "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"),
    ),
)
@pytest.mark.parametrize(
    argnames="scheduler_name",
    argvalues=(
        "euler",
        "euler_ancestral",
    ),
)
@pytest.mark.parametrize(
    argnames="cutout_model",
    argvalues=(
        "grabcut",
        "vit-matte",
    ),
)
# @pytest.mark.parametrize(
#     argnames="suffix",
#     argvalues=("", " on a white background"),
# )
@pytest.mark.parametrize(
    argnames="return_dict",
    argvalues=(
        True,
        False,
    ),
)
def test_pipeline_pixart_sigma(
    fg_prompt: str,
    disable_centering: bool,
    k_alpha_intensity: float,
    scheduler_name: str,
    negative_prompts: List[str],
    image_size: int,
    steps: int,
    cutout_model: CutoutModel,
    sure_fg_threshold: float,
    maybe_fg_threshold: float,
    maybe_bg_threshold: float,
    num_images: int,
    # suffix: str,
    seed: int,
    vit_matte_model_id: str,
    pipeline_id: str,
    denoiser_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    return_dict: bool,
):
    pipe = AlfiePixArtSigmaPipeline.from_alfie_setting(
        image_size=image_size,
        scheduler_name=scheduler_name,
        pipeline_id=pipeline_id,
        denoiser_id=denoiser_id,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)

    prompt_complete = ["A white background", fg_prompt]
    prompt = prompt_complete if not disable_centering else prompt_complete[1]

    output = pipe(
        prompt=prompt,
        nevative_prompt=negative_prompts,
        keep_cross_attention_maps=True,
        return_dict=return_dict,
        num_inference_steps=steps,
        centering=not disable_centering,
        generator=torch.Generator(device).manual_seed(seed),
    )
    images = output.images if return_dict else output[0]
    heatmaps = output.heatmaps if return_dict else output[1]

    assert len(images) == 1
    rgb_image = images[0]

    rgba_image = pipe.postprocess(
        rgb_image=rgb_image,
        heatmaps=heatmaps,
        cutout_model=cutout_model,
        image_size=image_size,
        sure_fg_threshold=sure_fg_threshold,
        maybe_fg_threshold=maybe_fg_threshold,
        maybe_bg_threshold=maybe_bg_threshold,
        k_alpha_intensity=k_alpha_intensity,
        vit_matte_model_id=vit_matte_model_id,
    )
    assert isinstance(rgba_image, PilImage)
