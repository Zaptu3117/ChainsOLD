from agents.artist.artist_lib import (
    AbtractPainter,
    Biomechanical, 
    Impressionist, 
    PixelArtist, 
    MultiMediaArtist, 
    PolaroidArtist, 
    SurrealistArtist   
)
from agents.critic.critic_lib import Critic
from agents.critic.critic_lib import (
    DrFusion, 
    DrLight, 
    DrPixel, 
    DrTerraNexus, 
    DrTheophilus, 
    DrWoolstonecraft
)

from random import randint

# We remove Biomechanical from the list, he is too cliché. # obsession of a topic is interresting. 
# Also, the abstract painting, make a break point into the narration. 
# Stratification and destratification. 
# Could we merge prompt ? Like making a matrix of meeting. 
# You have two style and you should combine them. 
# Pixel is a style attractor.
# Biomechanical is obsessed by some symbols.
# Abstract open new moments.
# 
#  Discussion pour sortir du cliché. 
artist_list = [AbtractPainter, 
               Impressionist, 
               Biomechanical,
               PixelArtist,
               MultiMediaArtist, 
               PolaroidArtist, 
               SurrealistArtist]

exhibition_name = input("Name of the exhibition : ")


def circular_exhibition(prompt, name_of_exhibition, width=1080, height=1920): 
    artist_random = randint(0, len(artist_list) - 1)
    artist = artist_list[artist_random]
    print("Artist name:", artist.__name__)
    artist_instance = artist()
    prompt = artist_instance.generate_artwork_prompt(message=description)
    print("\n Prompt:", prompt)
    path_image = artist_instance.create_image_sd3(prompt, exhibition_name, width, height)
    critic = Critic()
    description = critic.get_image_description(path_image)
    return description


if __name__ == "__main__": 
    initial_prompt = input("Initial prompt : ")
    exhibition_name = input("Exhibition name :")
    description = circular_exhibition(initial_prompt, exhibition_name)
    while True: 
        description = circular_exhibition(description, exhibition_name)


