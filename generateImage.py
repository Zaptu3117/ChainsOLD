# stable diffusion 3.
from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

import torch
from diffusers import StableDiffusion3Pipeline
import os
import re

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")


prompts = [
    "Balanced future city with both high-tech and natural elements coexisting",
    "Spectrum of possible futures, neither purely utopian nor dystopian",
    "Everyday life scene in a realistically imperfect but livable future",
    "Visual metaphor of navigating between extreme visions towards a middle path",
    "Futuristic problem-solving workshop tackling complex, nuanced challenges",
]


i = 0
for prompt in prompts:
    i += 1
    # Generate the image
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
        width=1024,
        height=1024,
    ).images[0]
    path = "UtopiaDystopia" + str(i) + ".png"
    image.save(path)
