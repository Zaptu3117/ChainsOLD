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


class Artist_pascal:
    def __init__(self) -> None:

        self.num_image = 0

    def create_image_sd3(
        self,
        prompt: str,
        objective_dec,
        litterary_description,
        bird_name: str = "random_exhibition",
        width=1024,
        height=1024,
    ) -> str:
        """
        General function that creates an images using Stable Diffusion 3.
        """

        width = (width // 64) * 64
        height = (height // 64) * 64

        bird_name = re.sub(r"[^\w\-_\. ]", "_", bird_name)
        bird_name = bird_name.strip().replace(" ", "_")

        exhibition_images_path = ""
        if not os.path.exists(os.getcwd() + "/pascal_img"):
            exhibition_images_path = os.mkdir(os.getcwd() + "/pascal_img")
        else:
            exhibition_images_path = os.getcwd() + "/pascal_img"
        if not os.path.exists(exhibition_images_path + "/" + bird_name):
            os.mkdir(exhibition_images_path + "/" + bird_name)

        # Generate the image
        image = pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0,
            width=width,
            height=height,
        ).images[0]

        # Create a valid filename from the prompt
        filename = re.sub(
            r"[^\w\-_\. ]", "_", prompt
        )  # Replace invalid filename characters
        filename = filename.strip().replace(" ", "_")  # Replace spaces with underscores
        filename = filename[:40]  # Limit filename length
        # Ensure the filename is unique
        base_filename = f"{filename}_{self.num_image}"
        self.num_image += 1
        counter = 1
        while os.path.exists(f"{exhibition_images_path}/{bird_name}/{filename}.png"):
            filename = f"{base_filename}_{counter}"
            counter += 1

        # Save the image
        exhibition_images_path = exhibition_images_path + f"/{bird_name}/"
        image_path = f"{exhibition_images_path}{filename}.png"
        image.save(image_path)

        with open(f"{exhibition_images_path}{filename}.txt", "w") as text_file:
            text_file.write("Objective description : ")
            text_file.write(objective_dec)
            text_file.write("\n")
            text_file.write("Litterary description : ")
            text_file.write(litterary_description)
            text_file.write("\n")
            text_file.write("The prompt :")
            text_file.write(prompt)

        return image_path

    @staticmethod
    def prompt_litterary(bird_name: str) -> str:
        prompt_description_scientific = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """When given a bird's scientific name, provide a detailed description of the species in no more than 100 words. Include information on:

        Physical appearance (size, shape, coloration, distinctive features)
        Plumage variations (if applicable for different sexes or seasons)
        Beak and feet characteristics
        Typical vocalizations or calls
        Preferred habitat
        Any unique behaviors or adaptations

        Ensure the description is vivid and precise, allowing the reader to visualize the bird. Avoid technical jargon unless necessary. Maintain a balance between scientific accuracy and engaging language.""",
                ),
                ("human", "The bird name : {bird_name}"),
            ]
        )

        prompt_description_litterary = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """After the factual description, provide a literary passage of 50-75 words that captures the essence of what this creature embodies, without mentioning the bird or any bird-specific terms. Focus on:

        The feeling of its habitat
        The mood its presence might inspire
        The sensations associated with its environment
        Abstract concepts it might represent (e.g., freedom, persistence, grace)
        Metaphors for its impact on the world around it

        Use evocative, sensory language to paint a picture of the intangible qualities and atmosphere associated with this being's existence, allowing the reader to connect emotionally without directly visualizing the creature itself.""",
                ),
                ("human", "This is the factual description : {factual_description}"),
            ]
        )

        fmt_prompt_scientific = prompt_description_scientific.invoke(
            {"bird_name": bird_name}
        )

        model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
        ans_objective_descrition = model.invoke(fmt_prompt_scientific).content
        print(ans_objective_descrition)

        fmt_prompt_litterary = prompt_description_litterary.invoke(
            {"factual_description": ans_objective_descrition}
        )
        ans_litterary_descrpiton = model.invoke(fmt_prompt_litterary).content
        print(ans_litterary_descrpiton)
        return ans_litterary_descrpiton, ans_objective_descrition

    @staticmethod
    def design_prompt(prompt_litterary: str) -> str:
        prompt_design_object = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Based on the literary description, create a prompt  for a text-to-image model. The prompt should describe an abstract painting that captures the essence of the described bird. Consider:

        The overall color palette inspired by the bird's plumage and habitat
        Key shapes or forms that abstractly represent the bird's features
        Textures or brush strokes that evoke the bird's characteristics
        The mood or emotion the painting should convey
        Potential abstract elements that symbolize the bird's behavior or habitat
        The composition and balance of the abstract artwork.
        Always start by "Abstract Painting style"

Just give back the prompt, no comment.
Ensure the prompt is vivid and specific, but does not directly mention the bird. Focus on creating an evocative abstract concept that captures the creature's spirit through color, form, and texture in a non-representational manner.
        """,
                ),
                ("human", "The literrary description : {litterary_description}"),
            ]
        )
        model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
        fmt_prompt_design_object = prompt_design_object.invoke(
            {"litterary_description": prompt_litterary}
        )
        ans_design_object = model.invoke(fmt_prompt_design_object).content
        print(ans_design_object)
        return ans_design_object
