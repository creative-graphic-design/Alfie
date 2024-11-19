from typing import List

import pytest
import torch

from alfie.pipelines import AlfiePixArtSigmaPipeline


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def torch_dtype() -> torch.dtype:
    return torch.bfloat16


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
def vit_matte_key() -> str:
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
    argnames="image_size, model_id",
    argvalues=(
        (256, "PixArt-alpha/PixArt-Sigma-XL-2-256x256"),
        (512, "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"),
    ),
)
@pytest.mark.parametrize(
    argnames="scheduler_name",
    argvalues=("euler", "euler_ancestral"),
)
@pytest.mark.parametrize(
    argnames="cutout_model",
    argvalues=("grabcut", "vit-matte"),
)
@pytest.mark.parametrize(
    argnames="suffix",
    argvalues=("", " on a white background"),
)
def test_pipeline_pixart_sigma(
    fg_prompt: str,
    disable_centering: bool,
    k_alpha_intensity: float,
    scheduler_name: str,
    negative_prompts: List[str],
    image_size: int,
    steps: int,
    cutout_model: str,
    sure_fg_threshold: float,
    maybe_fg_threshold: float,
    maybe_bg_threshold: float,
    num_images: int,
    suffix: str,
    seed: int,
    vit_matte_key: str,
    pipeline_id: str,
    model_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
):
    pipe = AlfiePixArtSigmaPipeline.from_alfie_setting(
        image_size=image_size, scheduler_name=scheduler_name, torch_dtype=torch_dtype
    )
    pipe = pipe.to(device)
    prompt_complete = ["A white background", fg_prompt]
    prompt_full = " ".join(prompt_complete[1].split())
    prompt = prompt_complete if not disable_centering else prompt_complete[1]
    prompt += suffix
