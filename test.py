from agents.artist.artist_lib import ChinesePainter, Impressionist, SurrealistArtist, OrientalPainter


list_artist = [
    ChinesePainter, 
    Impressionist, 
    SurrealistArtist, 
    OrientalPainter
]

original_prompt = "A woman"
exhibition_name = "WomanPainting"

for artist in list_artist:
    for i in range(3):  
        artist_inst = artist()
        prompt = artist_inst.generate_artwork_prompt(original_prompt)
        artist_inst.create_image_sd3(prompt, exhibition_name=exhibition_name)
