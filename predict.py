import os
from cog import BasePredictor, Input, Path
import sys
sys.path.append('/content/stable-cascade-hf')
os.chdir('/content/stable-cascade-hf')

import random
import numpy as np
import PIL.Image
import torch
from typing import List
from diffusers.utils import numpy_to_pil
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def generate(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    prior_num_inference_steps: int = 30,
    prior_guidance_scale: float = 4.0,
    decoder_num_inference_steps: int = 12,
    decoder_guidance_scale: float = 0.0,
    num_images_per_prompt: int = 2,
    prior_pipeline=None,
    decoder_pipeline=None,
) -> str:
    prior_pipeline.to("cuda:0")
    decoder_pipeline.to("cuda:0")
    generator = torch.Generator().manual_seed(seed)
    prior_output = prior_pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=prior_num_inference_steps,
        negative_prompt=negative_prompt,
        guidance_scale=prior_guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
    )

    # thanks to @DragonFarts ❤ for image_embeddings = result['image_embeddings']
    image_embeddings = None
    for result in prior_output:
        if 'image_embeddings' in result:
            image_embeddings = result['image_embeddings']
            break

    decoder_output = decoder_pipeline(
        image_embeddings=image_embeddings,
        prompt=prompt,
        num_inference_steps=decoder_num_inference_steps,
        guidance_scale=decoder_guidance_scale,
        negative_prompt=negative_prompt,
        generator=generator,
        output_type="pil",
    ).images

    image_path = "/tmp/image.png"
    decoder_output[0].save(image_path)
    return image_path

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.prior_pipeline = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16)
        self.decoder_pipeline = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade",  torch_dtype=torch.bfloat16)
        self.prior_pipeline.to("cuda:0")
        self.decoder_pipeline.to("cuda:0")
        # self.prior_pipeline.prior = torch.compile(self.prior_pipeline.prior, mode="reduce-overhead", fullgraph=True)
        # self.decoder_pipeline.decoder = torch.compile(self.decoder_pipeline.decoder, mode="max-autotune", fullgraph=True)
    def predict(
        self,
        prompt: str = Input(default=""),
        negative_prompt: str = Input(default=""),
        seed: int = Input(default=0),
        width: int = Input(default=1024),
        height: int = Input(default=1024),
        prior_num_inference_steps: int = Input(default=30),
        prior_guidance_scale: float = Input(default=4.0),
        decoder_num_inference_steps: int = Input(default=12),
        decoder_guidance_scale: float = Input(default=0.0),
        num_images_per_prompt: int = Input(default=2),
    ) -> Path:
        output_image = generate(prompt, negative_prompt, seed, width, height, prior_num_inference_steps, prior_guidance_scale, decoder_num_inference_steps, decoder_guidance_scale, num_images_per_prompt, self.prior_pipeline, self.decoder_pipeline)
        return Path(output_image)