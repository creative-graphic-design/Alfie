from typing import Callable, List, Optional, Sequence, Tuple, Union

import nltk
import torch
from diffusers.models import AutoencoderKL, PixArtTransformer2DModel
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
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
from transformers import T5EncoderModel, T5Tokenizer

from alfie.models import AlfieAttnProcessor2_0, AlfinePixArtTransformer2DModel
from alfie.utils import normalize_masks

logger = logging.get_logger(__name__)

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
    ) -> "AlfiePixArtSigmaPipeline":
        download_nltk_data()

        if image_size == 256:
            denoiser_id = "PixArt-alpha/PixArt-Sigma-XL-2-256x256"
        elif image_size == 512:
            denoiser_id = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
        else:
            raise ValueError(f"Invalid image size: {image_size}")

        transformer = AlfinePixArtTransformer2DModel.from_pretrained(
            denoiser_id,
            subfolder="transformer",
            use_safetensors=True,
            torch_dtype=torch.float16,
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
            pipeline_id, transformer=transformer, scheduler=eul_scheduler
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

    def parse_nouns(self, prompt: str, nouns_to_exclude=None):
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
        return nouns, nouns_indexes, num_prompt_tokens

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
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
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.FloatTensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            max_sequence_length (`int` defaults to 120): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
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

        device = self._execution_device  # THIS AUTOMATICALLY CHANGES TO CPU
        device = torch.device("cuda")

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        self.text_encoder = self.text_encoder.to(device)

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
        nouns, nouns_indexes, num_prompt_tokens = self.parse_nouns(
            prompt_clean, nouns_to_exclude=nouns_to_exclude
        )
        if len(nouns_indexes) == 0:
            logger.warning(
                f"No nouns found in the prompt {prompt_clean}. Returning None and skipping the generation."
            )
            return None, None, None

        self.text_encoder = self.text_encoder.to("cpu")
        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        self.set_progress_bar_config(disable=disable_tqdm)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            processor.num_prompt_tokens = num_prompt_tokens
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
                        return_dict=False,
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
        for noun in nouns:
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

        return ImagePipelineOutput(images=image)
