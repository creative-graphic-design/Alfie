from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import nltk
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL, PixArtTransformer2DModel
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_256_BIN,
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
)
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import (
    PixArtSigmaPipeline,
    retrieve_timesteps,
)
from diffusers.schedulers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    KarrasDiffusionSchedulers,
)
from diffusers.utils import deprecate, logging
from PIL import Image
from PIL.Image import Image as PilImage
from transformers import T5EncoderModel, T5Tokenizer

from alfie.models import AlfieAttnProcessor2_0, AlfinePixArtTransformer2DModel
from alfie.utils import normalize_masks

from .pipeline_output import AlfieHeatmaps, AlfieImagePipelineOutput

logger = logging.get_logger(__name__)

CutoutModel = Literal["grabcut", "vit-matte"]

ASPECT_RATIO_2048_BIN = {
    "0.25": [1024.0, 4096.0],
    "0.26": [1024.0, 3968.0],
    "0.27": [1024.0, 3840.0],
    "0.28": [1024.0, 3712.0],
    "0.32": [1152.0, 3584.0],
    "0.33": [1152.0, 3456.0],
    "0.35": [1152.0, 3328.0],
    "0.4": [1280.0, 3200.0],
    "0.42": [1280.0, 3072.0],
    "0.48": [1408.0, 2944.0],
    "0.5": [1408.0, 2816.0],
    "0.52": [1408.0, 2688.0],
    "0.57": [1536.0, 2688.0],
    "0.6": [1536.0, 2560.0],
    "0.68": [1664.0, 2432.0],
    "0.72": [1664.0, 2304.0],
    "0.78": [1792.0, 2304.0],
    "0.82": [1792.0, 2176.0],
    "0.88": [1920.0, 2176.0],
    "0.94": [1920.0, 2048.0],
    "1.0": [2048.0, 2048.0],
    "1.07": [2048.0, 1920.0],
    "1.13": [2176.0, 1920.0],
    "1.21": [2176.0, 1792.0],
    "1.29": [2304.0, 1792.0],
    "1.38": [2304.0, 1664.0],
    "1.46": [2432.0, 1664.0],
    "1.67": [2560.0, 1536.0],
    "1.75": [2688.0, 1536.0],
    "2.0": [2816.0, 1408.0],
    "2.09": [2944.0, 1408.0],
    "2.4": [3072.0, 1280.0],
    "2.5": [3200.0, 1280.0],
    "2.89": [3328.0, 1152.0],
    "3.0": [3456.0, 1152.0],
    "3.11": [3584.0, 1152.0],
    "3.62": [3712.0, 1024.0],
    "3.75": [3840.0, 1024.0],
    "3.88": [3968.0, 1024.0],
    "4.0": [4096.0, 1024.0],
}

NOUNS_TO_EXCLUDE: List[str] = [
    "image",
    "images",
    "picture",
    "pictures",
    "photo",
    "photograph",
    "photographs",
    "illustration",
    "paintings",
    "drawing",
    "drawings",
    "sketch",
    "sketches",
    "art",
    "arts",
    "artwork",
    "artworks",
    "poster",
    "posters",
    "cover",
    "covers",
    "collage",
    "collages",
    "design",
    "designs",
    "graphic",
    "graphics",
    "logo",
    "logos",
    "icon",
    "icons",
    "symbol",
    "symbols",
    "emblem",
    "emblems",
    "badge",
    "badges",
    "stamp",
    "stamps",
    "img",
    "video",
    "videos",
    "clip",
    "clips",
    "film",
    "films",
    "movie",
    "movies",
    "meme",
    "grand",
    "sticker",
    "stickers",
    "banner",
    "banners",
    "billboard",
    "billboards",
    "label",
    "labels",
    "scene",
    "art",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "www",
    "com",
    "net",
    "org",
    "http",
    "https",
    "html",
    "css",
    "js",
    "php",
    "scene",
    "view",
    "m3",
]


def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")


def grabcut(
    image,
    attention_maps,
    image_size,
    sure_fg_threshold,
    maybe_fg_threshold,
    maybe_bg_threshold,
):
    sure_fg_full_mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    maybe_fg_full_mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    maybe_bg_full_mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    sure_bg_full_mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    for attention_map in attention_maps:
        attention_map = F.interpolate(
            attention_map[None, None, :, :].float(),
            size=(image_size, image_size),
            mode="bicubic",
        )[0, 0]

        threshold_sure_fg = sure_fg_threshold * attention_map.max()
        threshold_maybe_fg = maybe_fg_threshold * attention_map.max()
        threshold_maybe_bg = maybe_bg_threshold * attention_map.max()
        sure_fg_full_mask += (attention_map > threshold_sure_fg).to(torch.uint8)
        maybe_fg_full_mask += (
            (attention_map > threshold_maybe_fg) & (attention_map <= threshold_sure_fg)
        ).to(torch.uint8)
        maybe_bg_full_mask += (
            (attention_map > threshold_maybe_bg) & (attention_map <= threshold_maybe_fg)
        ).to(torch.uint8)
        sure_bg_full_mask += (attention_map <= threshold_maybe_bg).to(torch.uint8)

    mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    mask = torch.where(sure_bg_full_mask.bool(), cv2.GC_BGD, mask)
    mask = torch.where(maybe_bg_full_mask.bool(), cv2.GC_PR_BGD, mask)
    mask = torch.where(maybe_fg_full_mask.bool(), cv2.GC_PR_FGD, mask)
    mask = torch.where(sure_fg_full_mask.bool(), cv2.GC_FGD, mask)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = mask.numpy().astype(np.uint8)
    try:
        mask, bgdModel, fgdModel = cv2.grabCut(
            img=np.array(image),
            mask=mask,
            rect=None,
            bgdModel=bgdModel,
            fgdModel=fgdModel,
            iterCount=5,
            mode=cv2.GC_INIT_WITH_MASK,
        )
    except Exception as err:
        logger.warning(
            f"Grabcut failed, using default mask and mode={cv2.GC_INIT_WITH_MASK}: %s",
            err,
            exc_info=True,
        )
        mask = np.zeros_like(mask)
        center_rect = (128, 128, 384, 384)
        mask, bgdModel, fgdModel = cv2.grabCut(
            np.array(image),
            mask,
            center_rect,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_RECT,
        )

    alpha = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    return alpha


def compute_trimap(
    attention_maps, image_size, sure_fg_threshold, maybe_bg_threshold
) -> torch.Tensor:
    def _compute_trimap(
        attention_maps,
        image_size: int,
        sure_fg_threshold: float,
        maybe_bg_threshold: float,
    ):
        breakpoint()

        tensor_size = (image_size, image_size)
        sure_fg_full_mask = torch.zeros(tensor_size, dtype=torch.uint8)
        unsure = torch.zeros(tensor_size, dtype=torch.uint8)
        sure_bg_full_mask = torch.zeros(tensor_size, dtype=torch.uint8)
        for attention_map in attention_maps:
            attention_map = attention_map[None, None, :, :].float()

            breakpoint()
            attention_map = F.interpolate(
                attention_map, size=tensor_size, mode="bicubic"
            )[0, 0]

            threshold_sure_fg = sure_fg_threshold * attention_map.max()
            threshold_maybe_bg = maybe_bg_threshold * attention_map.max()
            sure_fg_full_mask += attention_map > threshold_sure_fg
            unsure += (attention_map > maybe_bg_threshold) & (
                attention_map <= threshold_sure_fg
            )
            sure_bg_full_mask += attention_map <= threshold_maybe_bg

        mask = torch.zeros(tensor_size, dtype=torch.uint8)
        mask = torch.where(sure_bg_full_mask, 255, mask)
        mask = torch.where(unsure, 128, mask)
        mask = torch.where(sure_fg_full_mask, 0, mask)

        return mask

    if isinstance(attention_maps, list):
        masks = [
            _compute_trimap(
                attention_map, image_size, sure_fg_threshold, maybe_bg_threshold
            )
            for attention_map in attention_maps
        ]
        return torch.stack(masks)
    else:
        return _compute_trimap(
            attention_maps, image_size, sure_fg_threshold, maybe_bg_threshold
        )[None]


def combine_to_rgba_image(
    rgb_image: PilImage, alpha_image: Union[torch.Tensor, np.ndarray]
) -> PilImage:
    alpha_image *= 255
    if isinstance(alpha_image, torch.Tensor):
        alpha_image = alpha_image.detach().clone().cpu().numpy()
        alpha_image = alpha_image.clip(0, 255).astype(np.uint8)

    alpha_image_pl = Image.fromarray(alpha_image, mode="L")
    rgb_image = rgb_image.copy()
    rgb_image.putalpha(alpha_image_pl)
    return rgb_image


@dataclass
class ParsedNounts(object):
    nouns: List[str]
    nouns_indexes: List[List[int]]
    num_prompt_tokens: int


class AlfiePixArtSigmaPipeline(PixArtSigmaPipeline):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: PixArtTransformer2DModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)

    @classmethod
    def from_alfie_setting(
        cls,
        image_size: int,
        scheduler_name: str,
        torch_dtype: torch.dtype,
        pipeline_id: str = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        denoiser_id: Optional[str] = None,
    ) -> "AlfiePixArtSigmaPipeline":
        download_nltk_data()

        if image_size == 256:
            _denoiser_id = "PixArt-alpha/PixArt-Sigma-XL-2-256x256"
        elif image_size == 512:
            _denoiser_id = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
        else:
            raise ValueError(f"Invalid image size: {image_size}")

        if denoiser_id is not None and denoiser_id != _denoiser_id:
            raise ValueError(
                f"The image size (= {image_size}) and "
                f"the specified model ID (= {denoiser_id}) are different."
            )

        transformer = AlfinePixArtTransformer2DModel.from_pretrained(
            denoiser_id,
            subfolder="transformer",
            use_safetensors=True,
            torch_dtype=torch_dtype,
        )

        dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained(
            pipeline_id, subfolder="scheduler"
        )
        if scheduler_name == "euler":
            eul_scheduler = EulerDiscreteScheduler.from_config(
                config=dpm_scheduler.config,
            )
        elif scheduler_name == "euler_ancestral":
            eul_scheduler = EulerAncestralDiscreteScheduler.from_config(
                config=dpm_scheduler.config,
            )
        else:
            raise ValueError(f"Invalid scheduler: {scheduler_name}")

        return cls.from_pretrained(
            pipeline_id,
            transformer=transformer,
            scheduler=eul_scheduler,
            torch_dtype=torch_dtype,
        )

    def create_generation_mask(
        self,
        latent_h: int,
        latent_w: int,
        mask_type: str,
        dtype: torch.dtype,
        beta: float = 0.00,
        border_h: Optional[int] = None,
        border_w: Optional[int] = None,
    ) -> torch.Tensor:
        border_size_h = latent_h // 4 if border_h is None else border_h
        border_size_w = latent_w // 4 if border_w is None else border_w
        mask = torch.full((latent_h, latent_w), fill_value=beta).to(dtype)
        if mask_type == "center":
            mask[border_size_h:-border_size_h, border_size_w:-border_size_w] = 1
        elif mask_type == "bottom":
            mask[border_size_h * 2 :, border_size_w:-border_size_w] = 1
        elif mask_type == "top":
            mask[: -border_size_h * 2, border_size_w:-border_size_w] = 1
        elif mask_type == "right":
            mask[border_size_h:-border_size_h, border_size_w * 2 :] = 1
        elif mask_type == "left":
            mask[border_size_h:-border_size_h, : -border_size_w * 2] = 1
        elif mask_type == "top_right":
            mask[: -border_size_h * 2, border_size_w * 2 :] = 1
        elif mask_type == "top_left":
            mask[: -border_size_h * 2, : -border_size_w * 2] = 1
        elif mask_type == "bottom_right":
            mask[border_size_h * 2 :, border_size_w * 2 :] = 1
        elif mask_type == "bottom_left":
            mask[border_size_h * 2 :, : -border_size_w * 2] = 1
        elif mask_type == "full":
            mask = torch.ones((latent_h, latent_w)).to(dtype)
        else:
            raise ValueError(f"Invalid mask type: {mask_type}")
        return mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        clean_caption: bool = False,
        max_sequence_length: int = 120,
        **kwargs,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                PixArt-Alpha, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha, it's should be the embeddings of the ""
                string.
            clean_caption (`bool`, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            max_sequence_length (`int`, defaults to 120): Maximum sequence length to use for the prompt.
        """

        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # See Section 3.1. of the paper.
        max_length = max_sequence_length

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask
            prompt_attention_mask = prompt_attention_mask.to(device)

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device), attention_mask=prompt_attention_mask
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = [negative_prompt] * batch_size
            uncond_tokens = self._text_preprocessing(
                uncond_tokens, clean_caption=clean_caption
            )
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_prompt_attention_mask = uncond_input.attention_mask
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=negative_prompt_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                bs_embed, -1
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(
                num_images_per_prompt, 1
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def parse_nouns(
        self, prompt: str, nouns_to_exclude: Optional[Sequence[str]] = None
    ) -> ParsedNounts:
        if nouns_to_exclude is None:
            nouns_to_exclude = []
        prompt = prompt.lower()
        words = nltk.word_tokenize(prompt)
        nouns = [
            word
            for word, pos in nltk.pos_tag(words)
            if pos[:2] == "NN" and word not in nouns_to_exclude
        ]
        tokens = self.tokenizer.tokenize(prompt)
        num_prompt_tokens = len(tokens) + 1
        nouns_indexes = []
        for noun in nouns:
            search_tokens = self.tokenizer.tokenize(noun.lower())
            start_indexes = [
                x
                for x in range(len(tokens))
                if tokens[x : x + len(search_tokens)] == search_tokens
            ]
            merge_indexes = []
            for index in start_indexes:
                merge_indexes += [i + index for i in range(0, len(search_tokens))]
            nouns_indexes.append(merge_indexes)

        return ParsedNounts(
            nouns=nouns,
            nouns_indexes=nouns_indexes,
            num_prompt_tokens=num_prompt_tokens,
        )

    def _postprocess_gradcut(
        self,
        rgb_image: PilImage,
        heatmaps: AlfieHeatmaps,
        image_size: int,
        sure_fg_threshold: float,
        maybe_fg_threshold: float,
        maybe_bg_threshold: float,
        k_alpha_intensity: float,
    ) -> PilImage:
        alpha_mask = grabcut(
            image=rgb_image,
            attention_maps=list(heatmaps["cross_heatmaps_fg_nouns"].values()),
            image_size=image_size,
            sure_fg_threshold=sure_fg_threshold,
            maybe_fg_threshold=maybe_fg_threshold,
            maybe_bg_threshold=maybe_bg_threshold,
        )

        alpha_mask_alfie = torch.tensor(alpha_mask)
        alfa_hat = normalize_masks(
            heatmaps["ff_heatmap"] + 1 * heatmaps["cross_heatmap_fg"]
        )

        alfa_hat = (alfa_hat + k_alpha_intensity * alfa_hat).clip(0, 1)
        alpha_mask_alfie = torch.where(alpha_mask_alfie == 1, alfa_hat, 0.0)

        return combine_to_rgba_image(rgb_image=rgb_image, alpha_image=alpha_mask_alfie)

    def _postprocess_vit_matte(
        self,
        rgb_image: PilImage,
        heatmaps: AlfieHeatmaps,
        image_size: int,
        model_id: str,
        sure_fg_threshold: float,
        maybe_bg_threshold: float,
    ) -> PilImage:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        vit_matte_processor = VitMatteImageProcessor.from_pretrained(model_id)
        vit_matte_model = VitMatteForImageMatting.from_pretrained(model_id)
        vit_matte_model = vit_matte_model.eval()

        trimap = compute_trimap(
            attention_maps=[list(heatmaps["cross_heatmaps_fg_nouns"].values())],
            image_size=image_size,
            sure_fg_threshold=sure_fg_threshold,
            maybe_bg_threshold=maybe_bg_threshold,
        )

        vit_matte_inputs = vit_matte_processor(
            images=rgb_image, trimaps=trimap, return_tensors="pt"
        )
        vit_matte_inputs = vit_matte_inputs.to(self.device)
        vit_matte_model = vit_matte_model.to(self.device)

        with torch.no_grad():
            alpha_mask = vit_matte_model(**vit_matte_inputs).alphas[0, 0]
        alpha_mask = 1 - alpha_mask.detach().clone().cpu().numpy()

        return combine_to_rgba_image(rgb_image=rgb_image, alpha_image=alpha_mask)

    def postprocess(
        self,
        rgb_image: PilImage,
        heatmaps: AlfieHeatmaps,
        cutout_model: CutoutModel,
        image_size: int,
        sure_fg_threshold: float,
        maybe_fg_threshold: float,
        maybe_bg_threshold: float,
        k_alpha_intensity: float,
        vit_matte_model_id: str = "hustvl/vitmatte-base-composition-1k",
    ) -> PilImage:
        if cutout_model == "grabcut":
            return self._postprocess_gradcut(
                rgb_image=rgb_image,
                heatmaps=heatmaps,
                image_size=image_size,
                sure_fg_threshold=sure_fg_threshold,
                maybe_fg_threshold=maybe_fg_threshold,
                maybe_bg_threshold=maybe_bg_threshold,
                k_alpha_intensity=k_alpha_intensity,
            )
        elif cutout_model == "vit-matte":
            return self._postprocess_vit_matte(
                rgb_image=rgb_image,
                heatmaps=heatmaps,
                image_size=image_size,
                model_id=vit_matte_model_id,
                sure_fg_threshold=sure_fg_threshold,
                maybe_bg_threshold=maybe_bg_threshold,
            )
        else:
            raise ValueError(f"Invalid cutout model: {cutout_model}")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 300,
        keep_cross_attention_maps: bool = True,
        keep_self_attention_maps: bool = True,
        centering: bool = False,
        nouns_to_exclude: Sequence[str] = NOUNS_TO_EXCLUDE,
        disable_tqdm: bool = False,
        **kwargs,
    ) -> Union[AlfieImagePipelineOutput, Tuple]:
        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        if use_resolution_binning:
            if self.transformer.config.sample_size == 256:
                aspect_ratio_bin = ASPECT_RATIO_2048_BIN
            elif self.transformer.config.sample_size == 128:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_256_BIN
            else:
                raise ValueError("Invalid sample size")

            assert height is not None and width is not None
            orig_height, orig_width = height, width
            height, width = self.image_processor.classify_height_width_bin(
                height, width, ratios=aspect_ratio_bin
            )

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat(
            [negative_prompt_attention_mask, prompt_attention_mask], dim=0
        )

        foreground_prompt = prompt[1] if centering else prompt
        prompt_clean = self._text_preprocessing(
            foreground_prompt, clean_caption=clean_caption
        )[0]

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            1 * num_images_per_prompt,  # Modified
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        # 6.2 Modified
        latent_h, latent_w = latents.shape[-2:]
        mask_md = self.create_generation_mask(
            latent_h=latent_h,
            latent_w=latent_w,
            mask_type="center",
            dtype=prompt_embeds.dtype,
            beta=0.0,
            border_h=latent_h // 8,
            border_w=latent_w // 8,
        )

        processor = AlfieAttnProcessor2_0(
            keep_cross_attn_maps=keep_cross_attention_maps,
            keep_self_attn_maps=keep_self_attention_maps,
            tokenizer=self.tokenizer,
        )
        self.transformer.set_attn_processor(processor)
        parsed_nouns = self.parse_nouns(prompt_clean, nouns_to_exclude=nouns_to_exclude)
        if len(parsed_nouns.nouns_indexes) == 0:
            raise ValueError(
                f"No nouns found in the prompt {prompt_clean}. Returning None and skipping the generation."
            )

        self.text_encoder = self.text_encoder.to("cpu")
        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        self.set_progress_bar_config(disable=disable_tqdm)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            processor.num_prompt_tokens = parsed_nouns.num_prompt_tokens
            for i, t in enumerate(timesteps):
                processor.t = t
                processor.l_iteration_ca = 0
                processor.l_iteration_sa = 0

                with torch.no_grad():
                    latents = (
                        latents.repeat(batch_size, 1, 1, 1) if centering else latents
                    )
                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    current_timestep = t.clone()
                    if not torch.is_tensor(current_timestep):
                        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                        # This would be a good case for the `match` statement (Python 3.10+)
                        is_mps = latent_model_input.device.type == "mps"
                        if isinstance(current_timestep, float):
                            dtype = torch.float32 if is_mps else torch.float64
                        else:
                            dtype = torch.int32 if is_mps else torch.int64
                        current_timestep = torch.tensor(
                            [current_timestep],
                            dtype=dtype,
                            device=latent_model_input.device,
                        )
                    elif len(current_timestep.shape) == 0:
                        current_timestep = current_timestep[None].to(
                            latent_model_input.device
                        )
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    current_timestep = current_timestep.expand(
                        latent_model_input.shape[0]
                    )

                    # predict noise model_output
                    noise_pred = self.transformer(
                        latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=current_timestep,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=return_dict,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    # learned sigma
                    if self.transformer.config.out_channels // 2 == latent_channels:
                        noise_pred = noise_pred.chunk(2, dim=1)[0]
                    else:
                        noise_pred = noise_pred

                    # compute previous image: x_t -> x_t-1
                    pred = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=True
                    )
                    latents = pred["prev_sample"]
                    if centering:
                        latents = (
                            latents[0] * (1 - mask_md.cuda())
                            + latents[1] * mask_md.cuda()
                        ).unsqueeze(0)

                    if i == len(timesteps) - 1:
                        latents_pred_x0 = pred["pred_original_sample"].to(torch.float16)
                        decoded_x0 = self.vae.decode(
                            latents_pred_x0 / self.vae.config.scaling_factor,
                            return_dict=False,
                        )[0]
                        decoded_x0 = (decoded_x0 / 2 + 0.5).clamp(0, 1)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(
                    image, orig_width, orig_height
                )
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        heatmaps = {"cross_heatmaps_fg_nouns": {}, "self_heatmaps_fg_nouns": {}}

        # Compute foreground maps
        self_heatmap_fg = processor.compute_global_self_heat_map()
        cross_heatmap_fg = processor.compute_global_cross_heat_map(prompt=prompt_clean)
        cross_heatmaps_fg = []
        for noun in parsed_nouns.nouns:
            cross_heatmap_noun = cross_heatmap_fg.compute_word_heat_map(noun)
            self_heatmap_noun = self_heatmap_fg.compute_guided_heat_map(
                normalize_masks(cross_heatmap_noun.heatmap)
            )
            heatmaps["cross_heatmaps_fg_nouns"][noun] = normalize_masks(
                cross_heatmap_noun.expand_as(image[0])
            )
            heatmaps["self_heatmaps_fg_nouns"][noun] = normalize_masks(
                self_heatmap_noun.expand_as(image[0])
            )
            cross_heatmaps_fg.append(cross_heatmap_noun)

        cross_heatmaps_for_ff = normalize_masks(
            torch.stack([ca.heatmap for ca in cross_heatmaps_fg], dim=0).mean(dim=0)
        )
        ff = normalize_masks(
            self_heatmap_fg.compute_guided_heat_map(cross_heatmaps_for_ff).expand_as(
                image[0]
            )
        )
        cross_heatmaps_fg = normalize_masks(
            torch.stack(
                [ca.expand_as(image[0]) for ca in cross_heatmaps_fg], dim=0
            ).mean(dim=0)
        )
        heatmaps["cross_heatmap_fg"] = cross_heatmaps_fg
        heatmaps["ff_heatmap"] = ff

        if not return_dict:
            return image, heatmaps

        return AlfieImagePipelineOutput(images=image, heatmaps=heatmaps)
