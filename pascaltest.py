from pascal_helper import Artist_pascal

img_generator = Artist_pascal()

birds_names = ["Catherpes mexicanus",
"Syntheliboramphus craveri",
"Hylorchilus sumichrasti",
"Myiodynastes luteiven",
"Pitangus sulphuratus",
"Mydestes occidentalies Turbus albicolis", 
"Strix sartorii", 
"Penelopina nigra", 
"Sula nebouxii", 
"Psarocolius montezuma"]


for bird_name in birds_names:
    print(bird_name)
    litterary_description, objective_description = Artist_pascal.prompt_litterary(bird_name)
    print(litterary_description)
    print(objective_description)
    for i in range(10): 
        design_prompt = Artist_pascal.design_prompt(litterary_description)
        print(design_prompt)
        img_generator.create_image_sd3(design_prompt, objective_description, litterary_description, bird_name)
